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
import platform
from io import BytesIO
from typing import Final, Literal, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

LOGGER: Final = logging.getLogger(__name__)

#: Sentinel asking :func:`resolve_device` to detect rather than obey.
AUTO_DEVICE: Final = "auto"


#: What the device will be asked to do. The distinction is load-bearing on one
#: real configuration; see :func:`resolve_device`.
Workload = Literal["interactive", "batch"]


def _mps_is_usable() -> bool:
    """Report whether this torch build can actually run on MPS.

    ``is_available()`` alone is not enough: a wheel compiled without the backend
    still exposes ``torch.backends.mps``, and only ``is_built()`` separates "no
    backend in this build" from "no supported hardware".
    """
    import torch

    return bool(torch.backends.mps.is_available() and torch.backends.mps.is_built())


def _mps_shares_memory_with_cpu() -> bool:
    """Report whether MPS here means unified memory rather than a bus.

    Apple Silicon shares one memory pool between CPU and GPU, so moving a tensor
    to MPS costs almost nothing. An Intel Mac reaches MPS through a discrete or
    integrated AMD GPU across a real bus, where every forward pass pays a round
    trip a small workload cannot amortize.

    Architecture is the honest proxy for that difference, and it is the one
    torch does not expose directly.
    """
    return platform.machine() == "arm64"


def resolve_device(requested: str, *, workload: Workload = "interactive") -> str:
    """Pick the torch device to run the encoder on.

    Ordered by throughput — CUDA, then Apple's MPS, then CPU — with one measured
    exception described below. An accelerator that is *present but unusable*
    degrades to CPU rather than raising: a query encoder that fails at startup
    takes the whole application down in exchange for a speed-up nobody asked
    for, which is a bad trade at any corpus size.

    **Why ``workload`` exists.** Measured on the reference machine, an Intel Mac
    whose AMD GPU does in fact support MPS:

    ========================  ==========  ==========  =============
    workload                  CPU         MPS         faster
    ========================  ==========  ==========  =============
    one short text query      13.5 ms     50.6 ms     CPU, by 3.7x
    64-image batch            37 img/s    81 img/s    MPS, by 2.2x
    ========================  ==========  ==========  =============

    Both are real and neither generalises to the other. A single 77-token encode
    is dominated by the host-to-device round trip; a batch of images amortizes
    that trip over enough arithmetic to win decisively. Preferring MPS
    unconditionally would have made every search on this machine nearly four
    times slower in the name of acceleration.

    The split applies only where the round trip is a real cost — where CPU and
    GPU do not share memory. On Apple Silicon and on CUDA both workloads take
    the accelerator, which is what that hardware deserves. Encoded vectors are
    identical either way (measured cosine 1.000000), so this is a throughput
    decision and never a correctness one.

    Args:
        requested: ``"auto"`` to detect, or an explicit device name to return
            unchanged. An explicit name is obeyed without probing, so an
            operator can keep this process off a GPU that belongs to a training
            run — and can override the heuristic above in either direction.
        workload: ``"interactive"`` for one-off encodes on the request path,
            ``"batch"`` for the offline pass over a corpus.

    Returns:
        A device string safe to hand to ``SentenceTransformer``.
    """
    if requested != AUTO_DEVICE:
        return requested

    # Imported here rather than at module scope for the same reason `load`
    # defers `sentence_transformers`: importing this module must stay cheap.
    import torch

    if torch.cuda.is_available():
        return "cuda"

    if _mps_is_usable():
        if workload == "batch" or _mps_shares_memory_with_cpu():
            return "mps"
        LOGGER.info(
            "MPS is available but not used for interactive encoding on this "
            "architecture: without unified memory the transfer costs more than the "
            "forward pass saves. Set CORPUSLENS_TORCH_DEVICE=mps to override."
        )

    return "cpu"


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
    """CLIP bi-encoder over ``sentence-transformers``.

    Satisfies :class:`EmbeddingService`. Construct once per process — loading
    the checkpoint costs seconds — via :meth:`load` from the application
    lifespan, never per request.

    Runs wherever :func:`resolve_device` lands, which is CPU on the reference
    environment and an accelerator where one exists. Nothing else in the
    application needs to know which: the vectors are identical either way, so
    the device is a throughput decision, not a correctness one.
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
            device: Resolved torch device the model was placed on. Already
                concrete — :meth:`load` resolves ``"auto"`` before it gets here.
        """
        self._model = model
        self._device = device

    @property
    def device(self) -> str:
        """The resolved device this encoder runs on.

        Exposed so the lifespan can log what detection actually chose. A
        configured ``"auto"`` is not the answer to that question.
        """
        return self._device

    @classmethod
    def load(cls, model_id: str, device: str) -> ClipEmbeddingService:
        """Load the checkpoint and bind it to a device.

        Args:
            model_id: A ``sentence-transformers`` model id. It must be the
                checkpoint the corpus was embedded with — a mismatch yields a
                "shared" space that is not shared, and search degrades to noise
                rather than failing.
            device: Torch device, or ``"auto"`` to detect one. Resolution
                happens here rather than in the caller so that every entry
                point — the lifespan, a test, a future CLI — gets detection
                without repeating it.

        Returns:
            A service ready to encode.
        """
        # Imported here rather than at module scope so that importing this
        # module — which the Protocol above makes worth doing from anywhere —
        # does not drag in torch and ~2 s of import time.
        from sentence_transformers import SentenceTransformer

        resolved = resolve_device(device)
        LOGGER.info("Loading CLIP model %r on %s", model_id, resolved)
        return cls(SentenceTransformer(model_id, device=resolved), resolved)

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
