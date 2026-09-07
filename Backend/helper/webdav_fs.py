"""
Virtual filesystem for WebDAV backed by Telegram-Stremio MongoDB.

Layout (stable paths):

  /
  ├── Movies/
  │   └── Title (Year)/
  │       ├── Title (Year) - 1080p.mkv
  │       ├── Title (Year).nfo
  │       └── poster.jpg
  └── TV Shows/
      └── Show Name (Year)/
          ├── tvshow.nfo
          └── Season 01/
              ├── season.nfo
              ├── Show Name S01E01 - Episode Title - 1080p.mkv
              └── Show Name S01E01 - Episode Title.nfo
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from Backend import db
from Backend.helper.nfo_generator import episode_nfo, movie_nfo, season_nfo, tvshow_nfo
from Backend.logger import LOGGER


#----- sanitize a single path segment for Windows / media-server friendliness
_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_name(name: str, max_len: int = 120) -> str:
    s = _INVALID.sub("", (name or "").strip())
    s = re.sub(r"\s+", " ", s).strip(" .")
    if not s:
        s = "Unknown"
    return s[:max_len]


def movie_folder_name(doc: Dict[str, Any]) -> str:
    title = doc.get("title_english") or doc.get("title") or "Unknown"
    year = doc.get("release_year")
    base = safe_name(title)
    if year:
        return f"{base} ({year})"
    tmdb = doc.get("tmdb_id")
    if tmdb:
        return f"{base} {{tmdb-{tmdb}}}"
    return base


def show_folder_name(doc: Dict[str, Any]) -> str:
    return movie_folder_name(doc)


def quality_ext(name: str) -> str:
    n = (name or "").lower()
    for ext in (".mkv", ".mp4", ".avi", ".mov", ".m4v", ".webm", ".ts"):
        if n.endswith(ext):
            return ext
    return ".mkv"


def pick_best_quality(qualities: Optional[List[dict]]) -> Optional[dict]:
    if not qualities:
        return None
    order = {"2160p": 0, "4k": 0, "1440p": 1, "1080p": 2, "720p": 3, "480p": 4, "360p": 5}
    def key(q):
        ql = str(q.get("quality") or "").lower()
        return order.get(ql, 50)
    return sorted(qualities, key=key)[0]


def parse_size_bytes(size_str: Any, parts: Optional[List[dict]] = None) -> int:
    if parts:
        total = 0
        for p in parts:
            try:
                total += int(p.get("size_bytes") or 0)
            except (TypeError, ValueError):
                pass
        if total > 0:
            return total
    if isinstance(size_str, (int, float)):
        return int(size_str)
    if not size_str:
        return 0
    s = str(size_str).strip().upper().replace(",", "")
    m = re.match(r"^([\d.]+)\s*([KMGT]?B?)$", s)
    if not m:
        try:
            return int(float(s))
        except ValueError:
            return 0
    num = float(m.group(1))
    unit = m.group(2) or "B"
    mult = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
    return int(num * mult.get(unit, 1))


@dataclass
class VNode:
    """One virtual filesystem node."""
    path: str                          # absolute virtual path, no trailing slash (except root)
    name: str
    is_dir: bool
    size: int = 0
    mtime: float = field(default_factory=time.time)
    content_type: str = "application/octet-stream"
    # for files:
    kind: str = ""                     # movie_video | movie_nfo | show_nfo | season_nfo | episode_video | episode_nfo | poster
    # stream payload
    stream_id: Optional[str] = None    # QualityDetail.id (encoded stream hash)
    stream_name: Optional[str] = None
    parts: Optional[List[dict]] = None
    # nfo body (generated on demand if empty)
    nfo_body: Optional[bytes] = None
    # references back to DB
    media_type: Optional[str] = None
    tmdb_id: Optional[int] = None
    db_index: Optional[int] = None
    season_number: Optional[int] = None
    episode_number: Optional[int] = None
    children: Dict[str, "VNode"] = field(default_factory=dict)


@dataclass
class MediaIndexEntry:
    folder_name: str
    db_index: int
    doc_id: str
    tmdb_id: Optional[int] = None
    mtime: float = field(default_factory=time.time)
    vnode: Optional[VNode] = None


class WebDAVFilesystem:
    """
    High-performance virtual filesystem for WebDAV backed by Telegram-Stremio MongoDB.

    Architected for large libraries (10k+ movies, 50k+ episodes):
      1. Root (/) and Categories (/Movies, /TV Shows) resolve instantly with 0 DB overhead.
      2. Category directory listings query MongoDB using minimal field projections
         (title, year, tmdb_id), avoiding heavy document transfers.
      3. Folder contents (episodes, video qualities, NFOs) are generated lazily
         on-demand when the specific folder is accessed, and cached in an LRU.
      4. Concurrency lock prevents duplicate database scans (thundering herd).
      5. Periodic event loop yields prevent blocking other FastAPI requests.
    """

    def __init__(self, cache_ttl: int = 900, folder_lru_size: int = 500):
        self.cache_ttl = cache_ttl
        self.folder_lru_size = folder_lru_size

        self._lock = asyncio.Lock()
        self._movies_index: Dict[str, MediaIndexEntry] = {}  # lower_folder -> MediaIndexEntry
        self._shows_index: Dict[str, MediaIndexEntry] = {}   # lower_folder -> MediaIndexEntry
        self._movies_built_at: float = 0.0
        self._shows_built_at: float = 0.0

        # LRU cache for folder children: path -> Dict[name_lower, VNode]
        self._folder_cache: OrderedDict[str, Dict[str, VNode]] = OrderedDict()

        # Root and top-level static VNodes
        self._root_node = VNode(path="/", name="", is_dir=True)
        self._movies_node = VNode(path="/Movies", name="Movies", is_dir=True)
        self._shows_node = VNode(path="/TV Shows", name="TV Shows", is_dir=True)

    def invalidate(self) -> None:
        self._movies_built_at = 0.0
        self._shows_built_at = 0.0
        self._movies_index.clear()
        self._shows_index.clear()
        self._folder_cache.clear()

    async def ensure_tree(self) -> VNode:
        """Warm up movies and shows indexes."""
        await self._get_movies_index()
        await self._get_shows_index()
        return self._root_node

    def _cache_folder(self, path: str, children: Dict[str, VNode]) -> None:
        self._folder_cache[path] = children
        self._folder_cache.move_to_end(path)
        if len(self._folder_cache) > self.folder_lru_size:
            self._folder_cache.popitem(last=False)

    async def _get_movies_index(self) -> Dict[str, MediaIndexEntry]:
        now = time.time()
        if self._movies_index and (now - self._movies_built_at) < self.cache_ttl:
            return self._movies_index

        async with self._lock:
            if self._movies_index and (time.time() - self._movies_built_at) < self.cache_ttl:
                return self._movies_index

            LOGGER.info("[WebDAV] Indexing movies…")
            index: Dict[str, MediaIndexEntry] = {}
            storage_keys = sorted(
                [k for k in db.dbs.keys() if k.startswith("storage_")],
                key=lambda k: int(k.split("_")[1]),
            )
            count = 0
            for db_key in storage_keys:
                storage = db.dbs[db_key]
                try:
                    db_index = int(db_key.split("_")[1])
                except ValueError:
                    continue

                try:
                    # Projection: only fetch essential fields to keep payload tiny
                    cursor = storage["movie"].find(
                        {},
                        {
                            "_id": 1,
                            "title": 1,
                            "title_english": 1,
                            "release_year": 1,
                            "tmdb_id": 1,
                            "updated_at": 1,
                        },
                    )
                    async for doc in cursor:
                        doc_id = str(doc.get("_id"))
                        folder = movie_folder_name(doc)
                        base_folder = folder
                        n = 2
                        while folder.lower() in index:
                            folder = f"{base_folder} [{n}]"
                            n += 1

                        folder_path = f"/Movies/{folder}"
                        tmdb_id = doc.get("tmdb_id")
                        vnode = VNode(
                            path=folder_path,
                            name=folder,
                            is_dir=True,
                            media_type="movie",
                            tmdb_id=tmdb_id,
                            db_index=db_index,
                        )
                        entry = MediaIndexEntry(
                            folder_name=folder,
                            db_index=db_index,
                            doc_id=doc_id,
                            tmdb_id=tmdb_id,
                            vnode=vnode,
                        )
                        index[folder.lower()] = entry
                        count += 1
                        if count % 250 == 0:
                            await asyncio.sleep(0)  # Yield to event loop
                except Exception as e:
                    LOGGER.warning("[WebDAV] movie scan failed on %s: %s", db_key, e)

            self._movies_index = index
            self._movies_built_at = time.time()
            LOGGER.info("[WebDAV] Indexed %s movies", count)
            return self._movies_index

    async def _get_shows_index(self) -> Dict[str, MediaIndexEntry]:
        now = time.time()
        if self._shows_index and (now - self._shows_built_at) < self.cache_ttl:
            return self._shows_index

        async with self._lock:
            if self._shows_index and (time.time() - self._shows_built_at) < self.cache_ttl:
                return self._shows_index

            LOGGER.info("[WebDAV] Indexing TV shows…")
            index: Dict[str, MediaIndexEntry] = {}
            storage_keys = sorted(
                [k for k in db.dbs.keys() if k.startswith("storage_")],
                key=lambda k: int(k.split("_")[1]),
            )
            count = 0
            for db_key in storage_keys:
                storage = db.dbs[db_key]
                try:
                    db_index = int(db_key.split("_")[1])
                except ValueError:
                    continue

                try:
                    cursor = storage["tv"].find(
                        {},
                        {
                            "_id": 1,
                            "title": 1,
                            "title_english": 1,
                            "release_year": 1,
                            "tmdb_id": 1,
                            "updated_at": 1,
                        },
                    )
                    async for doc in cursor:
                        doc_id = str(doc.get("_id"))
                        folder = show_folder_name(doc)
                        base_folder = folder
                        n = 2
                        while folder.lower() in index:
                            folder = f"{base_folder} [{n}]"
                            n += 1

                        folder_path = f"/TV Shows/{folder}"
                        tmdb_id = doc.get("tmdb_id")
                        vnode = VNode(
                            path=folder_path,
                            name=folder,
                            is_dir=True,
                            media_type="tv",
                            tmdb_id=tmdb_id,
                            db_index=db_index,
                        )
                        entry = MediaIndexEntry(
                            folder_name=folder,
                            db_index=db_index,
                            doc_id=doc_id,
                            tmdb_id=tmdb_id,
                            vnode=vnode,
                        )
                        index[folder.lower()] = entry
                        count += 1
                        if count % 250 == 0:
                            await asyncio.sleep(0)  # Yield to event loop
                except Exception as e:
                    LOGGER.warning("[WebDAV] tv scan failed on %s: %s", db_key, e)

            self._shows_index = index
            self._shows_built_at = time.time()
            LOGGER.info("[WebDAV] Indexed %s TV shows", count)
            return self._shows_index

    async def _fetch_movie_doc(self, entry: MediaIndexEntry) -> Optional[dict]:
        storage = db.dbs.get(f"storage_{entry.db_index}")
        if not storage:
            return None
        doc = None
        try:
            from bson import ObjectId
            doc = await storage["movie"].find_one({"_id": ObjectId(entry.doc_id)})
        except Exception:
            pass
        if not doc:
            try:
                doc = await storage["movie"].find_one({"_id": entry.doc_id})
            except Exception:
                pass
        if doc:
            doc = _oid_str(doc)
            doc.setdefault("db_index", entry.db_index)
        return doc

    async def _fetch_show_doc(self, entry: MediaIndexEntry) -> Optional[dict]:
        storage = db.dbs.get(f"storage_{entry.db_index}")
        if not storage:
            return None
        doc = None
        try:
            from bson import ObjectId
            doc = await storage["tv"].find_one({"_id": ObjectId(entry.doc_id)})
        except Exception:
            pass
        if not doc:
            try:
                doc = await storage["tv"].find_one({"_id": entry.doc_id})
            except Exception:
                pass
        if doc:
            doc = _oid_str(doc)
            doc.setdefault("db_index", entry.db_index)
        return doc

    async def _get_movie_folder_children(self, entry: MediaIndexEntry) -> Dict[str, VNode]:
        folder_path = f"/Movies/{entry.folder_name}"
        if folder_path in self._folder_cache:
            return self._folder_cache[folder_path]

        doc = await self._fetch_movie_doc(entry)
        if not doc:
            return {}

        children: Dict[str, VNode] = {}
        # NFO
        nfo_name = f"{entry.folder_name}.nfo"
        try:
            nfo_bytes = movie_nfo(doc).encode("utf-8")
        except Exception as e:
            LOGGER.warning("[WebDAV] movie_nfo failed for %s: %s", entry.folder_name, e)
            nfo_bytes = b""

        children[nfo_name.lower()] = VNode(
            path=f"{folder_path}/{nfo_name}",
            name=nfo_name,
            is_dir=False,
            size=len(nfo_bytes),
            content_type="text/xml; charset=utf-8",
            kind="movie_nfo",
            nfo_body=nfo_bytes,
            media_type="movie",
            tmdb_id=doc.get("tmdb_id"),
            db_index=entry.db_index,
        )

        # Video qualities
        qualities = doc.get("telegram") or []
        for qual in qualities:
            video_node = self._movie_video_node(folder_path, entry.folder_name, doc, qual)
            if video_node and video_node.name.lower() not in children:
                children[video_node.name.lower()] = video_node

        self._cache_folder(folder_path, children)
        return children

    async def _get_show_folder_children(self, entry: MediaIndexEntry) -> Dict[str, VNode]:
        folder_path = f"/TV Shows/{entry.folder_name}"
        if folder_path in self._folder_cache:
            return self._folder_cache[folder_path]

        doc = await self._fetch_show_doc(entry)
        if not doc:
            return {}

        children: Dict[str, VNode] = {}
        # tvshow.nfo
        try:
            nfo_bytes = tvshow_nfo(doc).encode("utf-8")
        except Exception as e:
            LOGGER.warning("[WebDAV] tvshow_nfo failed for %s: %s", entry.folder_name, e)
            nfo_bytes = b""

        children["tvshow.nfo"] = VNode(
            path=f"{folder_path}/tvshow.nfo",
            name="tvshow.nfo",
            is_dir=False,
            size=len(nfo_bytes),
            content_type="text/xml; charset=utf-8",
            kind="show_nfo",
            nfo_body=nfo_bytes,
            media_type="tv",
            tmdb_id=doc.get("tmdb_id"),
            db_index=entry.db_index,
        )

        # Seasons
        for season in doc.get("seasons") or []:
            sn = int(season.get("season_number") or 0)
            season_name = f"Season {sn:02d}"
            season_path = f"{folder_path}/{season_name}"
            children[season_name.lower()] = VNode(
                path=season_path,
                name=season_name,
                is_dir=True,
                media_type="tv",
                tmdb_id=doc.get("tmdb_id"),
                db_index=entry.db_index,
                season_number=sn,
            )

        self._cache_folder(folder_path, children)
        return children

    async def _get_season_folder_children(self, entry: MediaIndexEntry, sn: int) -> Dict[str, VNode]:
        season_name = f"Season {sn:02d}"
        season_path = f"/TV Shows/{entry.folder_name}/{season_name}"
        if season_path in self._folder_cache:
            return self._folder_cache[season_path]

        doc = await self._fetch_show_doc(entry)
        if not doc:
            return {}

        season_data = next((s for s in (doc.get("seasons") or []) if int(s.get("season_number") or 0) == sn), None)
        if not season_data:
            return {}

        children: Dict[str, VNode] = {}
        # season.nfo
        try:
            snfo = season_nfo(doc, sn).encode("utf-8")
        except Exception as e:
            LOGGER.warning("[WebDAV] season_nfo failed: %s", e)
            snfo = b""

        children["season.nfo"] = VNode(
            path=f"{season_path}/season.nfo",
            name="season.nfo",
            is_dir=False,
            size=len(snfo),
            content_type="text/xml; charset=utf-8",
            kind="season_nfo",
            nfo_body=snfo,
            media_type="tv",
            tmdb_id=doc.get("tmdb_id"),
            db_index=entry.db_index,
            season_number=sn,
        )

        show_short = safe_name(doc.get("title_english") or doc.get("title") or "Show", 60)
        for ep in season_data.get("episodes") or []:
            en = int(ep.get("episode_number") or 0)
            ep_title = safe_name(ep.get("title") or f"Episode {en}", 80)
            qualities = ep.get("telegram") or []
            for qual in qualities:
                vnode = self._episode_video_node(
                    season_path, show_short, sn, en, ep_title, doc, ep, qual
                )
                if vnode and vnode.name.lower() not in children:
                    children[vnode.name.lower()] = vnode

            ep_nfo_name = f"{show_short} S{sn:02d}E{en:02d} - {ep_title}.nfo"
            try:
                ep_nfo_bytes = episode_nfo(doc, sn, ep).encode("utf-8")
            except Exception as e:
                ep_nfo_bytes = b""

            children[ep_nfo_name.lower()] = VNode(
                path=f"{season_path}/{ep_nfo_name}",
                name=ep_nfo_name,
                is_dir=False,
                size=len(ep_nfo_bytes),
                content_type="text/xml; charset=utf-8",
                kind="episode_nfo",
                nfo_body=ep_nfo_bytes,
                media_type="tv",
                tmdb_id=doc.get("tmdb_id"),
                db_index=entry.db_index,
                season_number=sn,
                episode_number=en,
            )

        self._cache_folder(season_path, children)
        return children

    async def resolve(self, path: str) -> Optional[VNode]:
        path = normalize_path(path)
        if path in ("", "/"):
            return self._root_node

        parts = [p for p in path.strip("/").split("/") if p]
        cat = parts[0].lower()

        # /Movies...
        if cat == "movies":
            if len(parts) == 1:
                return self._movies_node

            movies_idx = await self._get_movies_index()
            movie_folder = parts[1].lower()
            entry = movies_idx.get(movie_folder)
            if not entry:
                return None

            if len(parts) == 2:
                return entry.vnode

            if len(parts) == 3:
                filename = parts[2].lower()
                children = await self._get_movie_folder_children(entry)
                return children.get(filename)

            return None

        # /TV Shows...
        if cat in ("tv shows", "tvshows", "tv"):
            if len(parts) == 1:
                return self._shows_node

            shows_idx = await self._get_shows_index()
            show_folder = parts[1].lower()
            entry = shows_idx.get(show_folder)
            if not entry:
                return None

            if len(parts) == 2:
                return entry.vnode

            children = await self._get_show_folder_children(entry)
            p2 = parts[2].lower()

            if len(parts) == 3:
                return children.get(p2)

            if len(parts) == 4:
                m = re.match(r"^season\s*(\d+)$", p2)
                if not m:
                    return None
                sn = int(m.group(1))
                season_children = await self._get_season_folder_children(entry, sn)
                filename = parts[3].lower()
                return season_children.get(filename)

            return None

        return None

    async def list_dir(self, path: str) -> List[VNode]:
        path = normalize_path(path)
        if path in ("", "/"):
            return [self._movies_node, self._shows_node]

        parts = [p for p in path.strip("/").split("/") if p]
        cat = parts[0].lower()

        if cat == "movies":
            if len(parts) == 1:
                movies_idx = await self._get_movies_index()
                return [e.vnode for e in movies_idx.values() if e.vnode]

            movies_idx = await self._get_movies_index()
            entry = movies_idx.get(parts[1].lower())
            if not entry:
                return []
            if len(parts) == 2:
                children = await self._get_movie_folder_children(entry)
                return list(children.values())
            return []

        if cat in ("tv shows", "tvshows", "tv"):
            if len(parts) == 1:
                shows_idx = await self._get_shows_index()
                return [e.vnode for e in shows_idx.values() if e.vnode]

            shows_idx = await self._get_shows_index()
            entry = shows_idx.get(parts[1].lower())
            if not entry:
                return []

            if len(parts) == 2:
                children = await self._get_show_folder_children(entry)
                return list(children.values())

            if len(parts) == 3:
                p2 = parts[2].lower()
                m = re.match(r"^season\s*(\d+)$", p2)
                if not m:
                    return []
                sn = int(m.group(1))
                season_children = await self._get_season_folder_children(entry, sn)
                return list(season_children.values())

            return []

        return []

    def _movie_video_node(self, folder_path: str, folder: str, doc: dict, qual: dict) -> Optional[VNode]:
        qlabel = safe_name(str(qual.get("quality") or "Unknown"), 20)
        raw_name = qual.get("name") or folder
        ext = quality_ext(raw_name)
        fname = f"{folder} - {qlabel}{ext}"
        size = parse_size_bytes(qual.get("size"), qual.get("parts"))
        return VNode(
            path=f"{folder_path}/{fname}",
            name=fname,
            is_dir=False,
            size=size or 1,
            content_type=_mime_for_ext(ext),
            kind="movie_video",
            stream_id=qual.get("id"),
            stream_name=raw_name,
            parts=qual.get("parts"),
            media_type="movie",
            tmdb_id=doc.get("tmdb_id"),
            db_index=doc.get("db_index"),
        )

    def _episode_video_node(
        self,
        season_path: str,
        show_short: str,
        sn: int,
        en: int,
        ep_title: str,
        doc: dict,
        ep: dict,
        qual: dict,
    ) -> Optional[VNode]:
        qlabel = safe_name(str(qual.get("quality") or "Unknown"), 20)
        raw_name = qual.get("name") or f"{show_short}.S{sn:02d}E{en:02d}"
        ext = quality_ext(raw_name)
        fname = f"{show_short} S{sn:02d}E{en:02d} - {ep_title} - {qlabel}{ext}"
        size = parse_size_bytes(qual.get("size"), qual.get("parts"))
        return VNode(
            path=f"{season_path}/{fname}",
            name=fname,
            is_dir=False,
            size=size or 1,
            content_type=_mime_for_ext(ext),
            kind="episode_video",
            stream_id=qual.get("id"),
            stream_name=raw_name,
            parts=qual.get("parts"),
            media_type="tv",
            tmdb_id=doc.get("tmdb_id"),
            db_index=doc.get("db_index"),
            season_number=sn,
            episode_number=en,
        )


def normalize_path(path: str) -> str:
    if not path:
        return "/"
    path = path.replace("\\", "/")
    # decode is caller's job; here just clean
    while "//" in path:
        path = path.replace("//", "/")
    if not path.startswith("/"):
        path = "/" + path
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return path or "/"


def _mime_for_ext(ext: str) -> str:
    return {
        ".mkv": "video/x-matroska",
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".m4v": "video/x-m4v",
        ".webm": "video/webm",
        ".ts": "video/mp2t",
        ".nfo": "text/xml; charset=utf-8",
        ".jpg": "image/jpeg",
        ".png": "image/png",
    }.get(ext.lower(), "application/octet-stream")


def _oid_str(doc: dict) -> dict:
    from bson import ObjectId
    out = {}
    for k, v in doc.items():
        if isinstance(v, ObjectId):
            out[k] = str(v)
        else:
            out[k] = v
    return out


async def _async_sleep(sec: float) -> None:
    await asyncio.sleep(sec)


# singleton used by routes
fs = WebDAVFilesystem(cache_ttl=900)


def invalidate_webdav_cache() -> None:
    """Invalidate WebDAV filesystem cache and re-warm in background if preload is enabled."""
    fs.invalidate()
    try:
        from Backend.helper.settings_manager import SettingsManager
        if SettingsManager.current().webdav_preload:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(fs.ensure_tree())
            except RuntimeError:
                pass
    except Exception:
        pass
