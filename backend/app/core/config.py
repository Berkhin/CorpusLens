"""Environment-driven application settings.

Every path, port, model id and tunable the API needs is declared here so no
other module hardcodes one (CLAUDE.md §5.1). Values are read from the process
environment or the repository-root ``.env`` file, each prefixed ``CORPUSLENS_``.
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
    """Runtime configuration for the CorpusLens API.

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
        torch_device: Where the query encoder runs. ``"auto"`` resolves at
            startup via
            :func:`~app.services.embedding.resolve_device` (``cuda`` → ``mps``
            → ``cpu``). The explicit values are an operator override, which
            matters on a shared workstation whose GPU belongs to a training run
            this tool should stay off. A ``Literal`` rather than a free string
            so a typo in ``.env`` is a startup validation error naming the
            valid options, not an opaque torch failure on the first query.
        torch_num_threads: Cap on torch's intra-op CPU threads. Defaults to 1
            rather than to torch's own default of one-per-core, because this
            process runs forward passes *concurrently* from a worker pool: the
            two multiply, and 40 workers times 8 intra-op threads oversubscribe
            an 8-core machine forty-fold. Parallelism across requests is the
            useful axis here; parallelism within one short text encode is not.
            Raise it for a single-user machine running large batches; ``None``
            restores torch's default.
        worker_threads: Size of the anyio worker pool every blocking repository
            and encoder call is pushed onto. anyio's default of 40 assumes
            those threads do I/O and mostly wait; ours do CPU-bound torch and
            Lance work. Bounding the pool near the core count is what makes
            queueing, rather than thrashing, the behaviour under load. ``None``
            keeps anyio's default.
        search_nprobes: IVF partitions probed per query when an ANN index
            exists. Has almost no effect on recall (sweeping 1→256 on the
            reference corpus moved recall@20 by 0.008) because the loss under
            IVF-PQ is quantization, not pruning — see ``search_refine_factor``,
            which is the lever that matters, and CLAUDE.md §4.4.
        search_refine_factor: Multiplier on the candidate pool re-ranked against
            full-precision vectors before returning. This is what buys recall
            back: measured 0.695 → 0.997 at 10 on the reference corpus, and
            1.000 on a 200k-row corpus. Costs latency roughly linearly, which is
            a trade worth making — an approximate answer that looks exact is the
            failure mode this tool cannot afford.
        exact_scan_ceiling: Candidate count below which a *filtered* query
            bypasses the ANN index and scans exactly. Set from the measured cost
            of an exact scan (~1 ms per 1 000 rows), so the default is ~50 ms of
            work. The reason this exists is correctness, not speed: an IVF
            pre-filter applies within probed partitions, which measured 0.71
            recall against an exact 1.000 on a 20%-selective filter while still
            returning a full page. CLAUDE.md §4.4 has the table.
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
        env_prefix="CORPUSLENS_",
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
    torch_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    torch_num_threads: int | None = 1
    worker_threads: int | None = 8

    search_nprobes: int = 20
    search_refine_factor: int = 10
    exact_scan_ceiling: int = 50_000

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
