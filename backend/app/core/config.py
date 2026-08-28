"""Environment-driven application settings.

Every path, port, model id and tunable the API needs is declared here so no
other module hardcodes one (CLAUDE.md §5.1). Values are read from the process
environment or the repository-root ``.env`` file, each prefixed ``FLICKR8K_``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Final, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

#: Repository root, derived from this file's location:
#: ``backend/app/core/config.py`` → core → app → backend → root.
_REPO_ROOT: Final = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Runtime configuration for the Flickr8k explorer API.

    Attributes:
        host: Interface the server binds to. Defaults to loopback, not
            ``0.0.0.0``: this API has no authentication of any kind, and
            CLAUDE.md §2 scopes the whole tool to ``localhost``. Binding every
            interface would publish the corpus to the local network. The
            container image overrides it to ``0.0.0.0``, where the container
            boundary — not the bind address — is what limits reach.
        port: Port the server listens on. Read by ``python -m app``; the plain
            ``uvicorn`` CLI takes ``--port`` or ``UVICORN_PORT`` instead.
        data_dir: Root of the local data directory produced by
            ``scripts/ingest.py``. Everything else under ``data/`` is derived
            from it, so relocating the corpus is a single override.
        lancedb_uri: Explicit location of the LanceDB directory, overriding the
            default position under ``data_dir``. ``None`` keeps the two
            together, which is what the offline scripts assume; set it only to
            put the index on a different disk from the JPEGs.
        lancedb_table_name: Table written by the ingestion script.
        projection_file_name: Artefact written by ``scripts/project.py``. Its
            absence is not an error — the map view is simply unavailable, and
            the rest of the application serves normally.
        analysis_file_name: Artefact written by ``scripts/analyze.py``, optional
            on the same terms: without it the data-quality filters do not
            appear.
        collections_db_file_name: SQLite store holding user-created collections
            and the image overrides that re-partition the corpus. Unlike the two
            artefacts above this one is written by the API and created on first
            open, so its absence is a fresh install rather than a missing
            capability.
        max_collection_overrides: Ceiling on how many images may sit in a
            collection other than their own split, in total across the store —
            not per move, which bounds nothing, since the same total is
            reachable in batches. Every override becomes an id literal in each
            subsequent filtered query, and the cost is linear in it: measured on
            the real 8 000-row table, a filtered ``count_rows`` goes from 22 ms
            with none to 222 ms at this default and 1.6 s at a full
            re-partition. 1 000 is set where the query is still interactive and
            leaves generous room for the sets the overlay is *for* — the 32
            cross-split duplicates, the 200 weak captions, a holdout of a few
            hundred. Re-partitioning a whole corpus is a different operation and
            belongs in a new index; see ``docs/collections-next.md``. The full
            cost model is in
            :class:`~app.exceptions.CollectionMoveTooLargeError`.
        clip_model_id: Bi-encoder used for *text* queries. It must be the same
            checkpoint the images were embedded with — a mismatch silently
            yields a shared space that isn't shared, and search degrades to
            noise rather than failing loudly.
        torch_device: Pinned to ``"cpu"`` as a ``Literal`` rather than left a
            free string: CLAUDE.md §2 forbids CUDA/MPS, and torch 2.2.2 (the
            last macOS x86_64 wheel) has no usable accelerator here anyway.
            Typing it this way makes a misconfigured ``.env`` a startup
            validation error instead of a runtime crash.
        torch_num_threads: Cap on torch's intra-op CPU threads. ``None`` lets
            torch choose. Worth lowering if the API shares the machine with
            the dev server.
        cors_allow_origins: Browser origins permitted to call the API.
            Defaults to Vite's dev server on both loopback spellings. Accepts
            either a comma-separated list or a JSON array from the environment.
        images_url_prefix: Mount point for the static image files.
        api_prefix: Common prefix for all JSON endpoints.
        frontend_dist_dir: Compiled SPA to serve from the application root.
            ``None`` in development, where Vite serves the client on its own
            port and this process is API-only. The container image sets it, so
            one port serves both and no CORS hop is involved.
        log_level: Root log level name.
    """

    model_config = SettingsConfigDict(
        env_prefix="FLICKR8K_",
        env_file=_REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    host: str = "127.0.0.1"
    port: int = 8000

    data_dir: Path = _REPO_ROOT / "data"
    lancedb_uri: Path | None = None
    lancedb_table_name: str = "images"
    projection_file_name: str = "projection.json"
    analysis_file_name: str = "analysis.json"
    collections_db_file_name: str = "collections.db"

    max_collection_overrides: int = 1000

    clip_model_id: str = "clip-ViT-B-32"
    torch_device: Literal["cpu"] = "cpu"
    torch_num_threads: int | None = None

    #: ``NoDecode`` suppresses pydantic-settings' default JSON parse of complex
    #: types, which would reject the comma-separated spelling every other tool
    #: uses for an origin list with an opaque ``SettingsError``. The validator
    #: below accepts both forms instead.
    cors_allow_origins: Annotated[tuple[str, ...], NoDecode] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    )

    images_url_prefix: str = "/images"
    api_prefix: str = "/api"
    frontend_dist_dir: Path | None = None
    log_level: str = "INFO"

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _parse_origins(cls, value: object) -> object:
        """Accept an origin list as either JSON or a comma-separated string.

        Args:
            value: The raw environment value, or an already-structured default.

        Returns:
            A sequence of origins for the field's own validation to check, or
            the input untouched when it did not arrive as a string.
        """
        if not isinstance(value, str):
            return value
        text = value.strip()
        if text.startswith("["):
            return json.loads(text)
        return [origin.strip() for origin in text.split(",") if origin.strip()]

    @property
    def images_dir(self) -> Path:
        """Directory of original JPEGs, served statically."""
        return self.data_dir / "images"

    @property
    def lancedb_dir(self) -> Path:
        """Directory backing the embedded LanceDB database.

        Sits under :attr:`data_dir` unless :attr:`lancedb_uri` names another
        location outright.
        """
        if self.lancedb_uri is not None:
            return self.lancedb_uri
        return self.data_dir / "lancedb"

    @property
    def projection_path(self) -> Path:
        """File holding the pre-computed 2-D coordinates, if one was built."""
        return self.data_dir / self.projection_file_name

    @property
    def analysis_path(self) -> Path:
        """File holding the data-quality measurements, if they were computed."""
        return self.data_dir / self.analysis_file_name

    @property
    def collections_path(self) -> Path:
        """File backing the user-collection overlay store.

        Sits under ``data/`` like the other artefacts, which is already
        gitignored wholesale and already the container's bind mount, so
        collections survive ``docker compose build`` with no compose change.
        SQLite's ``-wal`` and ``-shm`` sidecars land beside it.
        """
        return self.data_dir / self.collections_db_file_name


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that ``.env`` is parsed once and every ``Depends(get_settings)``
    resolves to the same immutable object.
    """
    return Settings()
