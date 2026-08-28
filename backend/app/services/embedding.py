"""The contract for projecting text and images into a shared vector space.

Split from :mod:`app.services.search_service` so that ranking logic and model
mechanics are separately replaceable: search cares that it receives a
unit-length vector in the corpus's space, not that a ``SentenceTransformer``
produced it. Only :class:`ClipEmbeddingService` below imports
``sentence_transformers``, which makes swapping the encoder — for a different
CLIP checkpoint, an ONNX export, or a remote service in a fork that permits one
— a change to this module alone.

**Both methods are blocking.** A forward pass is CPU work; keeping the contract
synchronous lets the service layer decide how to offload it (it uses
``anyio.to_thread``) and keeps this module free of any async framework, per
CLAUDE.md §4.1.

**On the two methods costing wildly different amounts.** :meth:`embed_text` is
the single exception CLAUDE.md §2 grants to "embedding is an offline batch job":
a short string is a ~tens-of-milliseconds pass over a handful of tokens.
:meth:`embed_image` is not that, and nothing in the request path calls it — see
its own docstring.
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Final, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

LOGGER: Final = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingService(Protocol):
    """Projects queries into the space the corpus was embedded in.

    Implementations must return **unit-length** vectors: the store ranks by
    cosine, and the ingestion script normalized the image side, so an
    unnormalized query does not fail — it quietly skews every ranking.

    ``runtime_checkable`` on the same terms as
    :class:`~app.repositories.vector_db.VectorRepository`: ``isinstance``
    confirms the methods exist, never that they have the right signature.
    """

    def embed_text(self, text: str) -> NDArray[np.float32]:
        """Project a text query into the shared space.

        Args:
            text: The query to embed; callers validate length beforehand.

        Returns:
            A 1-D float32 unit vector of the corpus's dimensionality.
        """
        ...

    def embed_image(self, image_bytes: bytes) -> NDArray[np.float32]:
        """Project a raw image into the shared space.

        Args:
            image_bytes: Encoded image in any format Pillow can open.

        Returns:
            A 1-D float32 unit vector of the corpus's dimensionality.

        Raises:
            ValueError: If the bytes are not a decodable image.
        """
        ...


class ClipEmbeddingService:
    """CLIP bi-encoder over ``sentence-transformers``, on the CPU.

    Satisfies :class:`EmbeddingService`. Construct once per process — loading
    the checkpoint costs seconds — via :meth:`load` from the application
    lifespan, never per request.
    """

    def __init__(self, model: object, device: str) -> None:
        """Wrap an already-loaded bi-encoder.

        Prefer :meth:`load`; this constructor exists so a test can inject a
        double without a checkpoint on disk.

        Args:
            model: A loaded ``SentenceTransformer``. Typed ``object`` because
                ``sentence_transformers`` ships no ``py.typed`` marker, so the
                real class resolves to ``Any`` and would silently disable
                checking on every attribute reached through it; the two call
                sites below narrow it explicitly instead.
            device: Torch device for the forward pass; always ``"cpu"`` here,
                per CLAUDE.md §2.
        """
        self._model = model
        self._device = device

    @classmethod
    def load(cls, model_id: str, device: str) -> ClipEmbeddingService:
        """Load the checkpoint and bind it to a device.

        Args:
            model_id: A ``sentence-transformers`` model id. It must be the
                checkpoint the corpus was embedded with — a mismatch yields a
                "shared" space that is not shared, and search degrades to noise
                rather than failing.
            device: Torch device for the forward pass.

        Returns:
            A service ready to encode.
        """
        # Imported here rather than at module scope so that importing this
        # module — which the Protocol above makes worth doing from anywhere —
        # does not drag in torch and ~2 s of import time.
        from sentence_transformers import SentenceTransformer

        LOGGER.info("Loading CLIP model %r on %s", model_id, device)
        return cls(SentenceTransformer(model_id, device=device), device)

    def embed_text(self, text: str) -> NDArray[np.float32]:
        """Project a text query into CLIP's shared image/text space.

        ``normalize_embeddings=True`` is **not** the sentence-transformers
        default (verified against the installed 5.7.0 signature) and is
        load-bearing twice over: it is what the ingestion script applied to the
        image side, and cosine ranking is only meaningful when both sides are
        unit length. Dropping it would not raise — it would skew every ranking.

        ``encode`` is used rather than 5.x's ``encode_query`` deliberately: the
        latter applies a retrieval prompt template that CLIP has no notion of,
        and the images were embedded with plain ``encode``. Both sides of a
        shared space must be produced the same way.

        Args:
            text: The text to embed.

        Returns:
            A ``(512,)`` float32 unit vector.
        """
        return self._encode(text)

    def embed_image(self, image_bytes: bytes) -> NDArray[np.float32]:
        """Project a raw image into CLIP's shared image/text space.

        Goes through the *same* ``encode`` entry point as :meth:`embed_text`;
        sentence-transformers' CLIP wrapper dispatches on the argument type, and
        using one call path is what keeps both sides of the space consistent
        with how ``scripts/ingest.py`` built the index.

        **No request handler calls this, and none should.** A CPU forward pass
        over an image is ~100 ms here against ~10 ms for a string, and
        CLAUDE.md §2 confines image embedding to the offline pipeline. It exists
        because the interface is incomplete without it — an alternative encoder
        must be able to declare it, and a fork adding upload-to-search needs a
        defined place to call. Search-by-example in this API takes an
        ``image_id`` and reads the vector computed during ingestion instead,
        which costs no inference at all.

        Args:
            image_bytes: Encoded image in any format Pillow can open.

        Returns:
            A ``(512,)`` float32 unit vector.

        Raises:
            ValueError: If the bytes are not a decodable image.
        """
        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(BytesIO(image_bytes)) as handle:
                # RGB because CLIP's preprocessing expects three channels;
                # greyscale and palettised inputs would otherwise fail deeper in
                # the stack with a shape error naming nothing useful.
                image = handle.convert("RGB")
        except (UnidentifiedImageError, OSError) as error:
            raise ValueError("Input bytes are not a decodable image") from error

        return self._encode(image)

    def _encode(self, subject: object) -> NDArray[np.float32]:
        """Run the forward pass and return a float32 unit vector.

        Args:
            subject: A string or a ``PIL.Image.Image``; the model dispatches.

        Returns:
            A 1-D float32 unit vector.
        """
        # `SentenceTransformer` is untyped, so the call is narrowed here rather
        # than letting `Any` propagate into the return value.
        embedding = cast(
            NDArray[np.float32],
            self._model.encode(  # type: ignore[attr-defined] # untyped third-party
                subject,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                device=self._device,
            ),
        )
        return embedding.astype(np.float32, copy=False)
