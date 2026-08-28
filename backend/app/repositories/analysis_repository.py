"""The only module that reads the data-quality artefact.

``scripts/analyze.py`` writes ``data/analysis.json``; this reads it once at
startup. Like the projection, it is **optional**: without it the application
serves normally and the quality filters simply do not appear.

The artefact holds two kinds of thing, and they are exposed differently because
they are used differently. Per-image facts (nearest neighbour, caption rank) are
looked up one at a time by the detail and export paths. Corpus-level facts
(recall, the duplicate pair list) are read whole, and the *id sets* derived from
them are precomputed here rather than on each request: they never change, and
recomputing "which images are in a cross-split duplicate pair" per request would
be work with a constant answer.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from app.models.domain import CaptionRetrieval, DuplicatePair, ImageAnalysis, QualityFlag

LOGGER: Final = logging.getLogger(__name__)

_IMAGES_KEY: Final = "images"
_PAIRS_KEY: Final = "duplicate_pairs"
_RETRIEVAL_KEY: Final = "caption_retrieval"
_THRESHOLD_KEY: Final = "duplicate_threshold"

#: Size of the "weakest annotations" set, as a fraction of the corpus. The
#: filter exists to produce a review queue, so it wants a bounded worst-of list
#: rather than a threshold on a rank whose scale depends on corpus size.
_WEAK_CAPTION_FRACTION: Final = 0.025


class AnalysisRepository:
    """In-memory access to the data-quality measurements."""

    def __init__(
        self,
        *,
        images: Mapping[str, ImageAnalysis],
        duplicate_pairs: Sequence[DuplicatePair],
        duplicate_threshold: float,
        caption_retrieval: CaptionRetrieval | None,
    ) -> None:
        """Hold an already-parsed analysis and derive its id sets.

        Args:
            images: Per-image measurements, keyed by id.
            duplicate_pairs: Every pair above the threshold.
            duplicate_threshold: Cosine the pairs were selected with.
            caption_retrieval: Corpus recall, or ``None`` if the caption pass
                was skipped.
        """
        self._images = images
        self._duplicate_pairs = duplicate_pairs
        self._duplicate_threshold = duplicate_threshold
        self._caption_retrieval = caption_retrieval

        self._near_duplicate_ids = frozenset(
            image_id for pair in duplicate_pairs for image_id in (pair.a, pair.b)
        )
        self._cross_split_ids = frozenset(
            image_id
            for pair in duplicate_pairs
            if pair.cross_split
            for image_id in (pair.a, pair.b)
        )
        self._weak_caption_ids = self._derive_weak_caption_ids(images)
        self._caption_ranks: Mapping[str, int] = {
            image_id: analysis.caption_rank
            for image_id, analysis in images.items()
            if analysis.caption_rank is not None
        }

    @staticmethod
    def _derive_weak_caption_ids(images: Mapping[str, ImageAnalysis]) -> frozenset[str]:
        """Pick the images whose own captions retrieve them worst.

        A fraction of the corpus rather than a rank cutoff: "rank worse than
        100" means something different in a corpus of 8 000 than in one of
        80 000, whereas "the worst 2.5%" is a review queue of a predictable
        size either way.
        """
        ranked = [
            (image_id, analysis.caption_rank)
            for image_id, analysis in images.items()
            if analysis.caption_rank is not None
        ]
        if not ranked:
            return frozenset()
        ranked.sort(key=lambda entry: entry[1] or 0, reverse=True)
        size = max(1, round(len(images) * _WEAK_CAPTION_FRACTION))
        return frozenset(image_id for image_id, _ in ranked[:size])

    @classmethod
    def load(cls, path: Path) -> AnalysisRepository | None:
        """Read an analysis from disk.

        Args:
            path: Location of the artefact.

        Returns:
            The loaded repository, or ``None`` when the file is absent or
            unusable. A malformed file is logged and treated as absent: the
            failure belongs to the offline pipeline, and refusing to serve the
            gallery over it would punish the wrong thing.
        """
        if not path.is_file():
            LOGGER.info("No analysis at %s — quality filters will be unavailable", path)
            return None

        try:
            document: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("Could not read the analysis at %s; disabling quality filters", path)
            return None

        if not isinstance(document, dict) or not isinstance(document.get(_IMAGES_KEY), dict):
            LOGGER.warning("Analysis at %s has no images object; disabling quality filters", path)
            return None

        images = {
            str(image_id): ImageAnalysis(
                nearest_neighbour_id=str(entry.get("nn_id", "")),
                nearest_neighbour_similarity=float(entry.get("nn_similarity", 0.0)),
                caption_rank=(
                    int(entry["caption_rank"]) if entry.get("caption_rank") is not None else None
                ),
            )
            for image_id, entry in document[_IMAGES_KEY].items()
            if isinstance(entry, dict)
        }

        pairs = [
            DuplicatePair(
                a=str(entry["a"]),
                b=str(entry["b"]),
                a_split=str(entry.get("a_split", "")),
                b_split=str(entry.get("b_split", "")),
                similarity=float(entry.get("similarity", 0.0)),
                cross_split=bool(entry.get("cross_split", False)),
            )
            for entry in document.get(_PAIRS_KEY, [])
            if isinstance(entry, dict) and "a" in entry and "b" in entry
        ]

        retrieval_document = document.get(_RETRIEVAL_KEY)
        retrieval = (
            CaptionRetrieval(
                recall_at_1=float(retrieval_document.get("r_at_1", 0.0)),
                recall_at_5=float(retrieval_document.get("r_at_5", 0.0)),
                recall_at_10=float(retrieval_document.get("r_at_10", 0.0)),
                captions=int(retrieval_document.get("captions", 0)),
            )
            if isinstance(retrieval_document, dict)
            else None
        )

        LOGGER.info(
            "Loaded analysis for %d image(s) from %s: %d duplicate pair(s), %d crossing a split",
            len(images),
            path,
            len(pairs),
            sum(1 for pair in pairs if pair.cross_split),
        )
        return cls(
            images=images,
            duplicate_pairs=pairs,
            duplicate_threshold=float(document.get(_THRESHOLD_KEY, 0.0)),
            caption_retrieval=retrieval,
        )

    @property
    def duplicate_pairs(self) -> Sequence[DuplicatePair]:
        """Every near-duplicate pair, most similar first."""
        return self._duplicate_pairs

    @property
    def duplicate_threshold(self) -> float:
        """Cosine above which a pair was recorded."""
        return self._duplicate_threshold

    @property
    def caption_retrieval(self) -> CaptionRetrieval | None:
        """Corpus recall-at-k, or ``None`` when the caption pass was skipped."""
        return self._caption_retrieval

    @property
    def cross_split_pair_count(self) -> int:
        """How many pairs straddle a split boundary."""
        return sum(1 for pair in self._duplicate_pairs if pair.cross_split)

    @property
    def caption_ranks(self) -> Mapping[str, int]:
        """Each measured image's median own-caption rank against the corpus.

        Exposed as a mapping rather than looked up per image because the caller
        that needs it — re-aggregating recall over a collection — needs all of
        them at once, and 8 000 :meth:`get` calls to read one field each would
        be a loop where a projection will do.

        Derived once here rather than per request: like the id sets above, the
        answer is constant for the life of the process.

        Returns:
            Image id to rank, omitting images with no measured rank (which is
            every image when ``analyze.py --no-captions`` produced the file).
        """
        return self._caption_ranks

    def get(self, image_id: str) -> ImageAnalysis | None:
        """Return one image's measurements, or ``None`` if it has none."""
        return self._images.get(image_id)

    def ids_for(self, flag: QualityFlag) -> frozenset[str]:
        """Resolve a quality flag to the ids it selects.

        Precomputed at load time: the sets never change, so deriving them per
        request would be work with a constant answer.

        ``weak-captions`` is empty when the caption pass was skipped. That is
        deliberate — the filter then matches nothing, which reads as "this
        measurement does not exist here" rather than silently behaving as
        though no filter had been asked for.

        Args:
            flag: The finding to select.

        Returns:
            Ids satisfying it, possibly empty.
        """
        if flag == "near-duplicate":
            return self._near_duplicate_ids
        if flag == "cross-split-duplicate":
            return self._cross_split_ids
        return self._weak_caption_ids
