"""Device detection, exercised without depending on this machine's hardware.

``resolve_device`` is the one piece of this codebase whose correct answer
differs per machine, which makes it exactly the piece that cannot be tested by
running it and looking. Every test here fakes the hardware and asserts the
*decision*, so the suite gives the same verdict on a CUDA box, an Apple Silicon
laptop and the Intel Mac these rules were measured on.

The interesting case is the last one. Measured on the reference machine, MPS is
3.7x slower than CPU for a single short text encode and 2.2x faster for a
64-image batch, so the workload — not just the hardware — decides.
"""

from __future__ import annotations

import pytest

from app.services import embedding
from app.services.embedding import resolve_device


@pytest.fixture
def no_accelerator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Present a machine with neither CUDA nor MPS."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(embedding, "_mps_is_usable", lambda: False)


def _offer_mps(monkeypatch: pytest.MonkeyPatch, *, unified_memory: bool) -> None:
    """Present a machine with usable MPS and no CUDA.

    Args:
        monkeypatch: Fixture used to install the fakes.
        unified_memory: Whether to claim Apple Silicon, where a transfer to the
            GPU is nearly free, or an Intel Mac, where it is not.
    """
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(embedding, "_mps_is_usable", lambda: True)
    monkeypatch.setattr(embedding, "_mps_shares_memory_with_cpu", lambda: unified_memory)


@pytest.mark.parametrize("explicit", ["cpu", "cuda", "mps"])
def test_an_explicit_device_is_obeyed_without_probing(
    explicit: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator override wins over anything detection would conclude.

    This is what lets someone keep the API off a GPU that belongs to a training
    run — and what lets them overrule the interactive-MPS heuristic in either
    direction. It must not consult the hardware at all, so the probes are made
    to fail loudly if it does.
    """
    import torch

    def explode() -> bool:
        raise AssertionError("detection ran despite an explicit device")

    monkeypatch.setattr(torch.cuda, "is_available", explode)
    monkeypatch.setattr(embedding, "_mps_is_usable", explode)

    assert resolve_device(explicit) == explicit


def test_cuda_wins_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA is preferred for both workloads and needs no unified-memory caveat."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto", workload="interactive") == "cuda"
    assert resolve_device("auto", workload="batch") == "cuda"


def test_falls_back_to_cpu_with_no_accelerator(no_accelerator: None) -> None:
    """A machine with nothing available gets CPU rather than an exception.

    Degrading is the whole contract: a query encoder that raises at startup
    takes the application down for a speed-up nobody asked for.
    """
    assert resolve_device("auto", workload="interactive") == "cpu"
    assert resolve_device("auto", workload="batch") == "cpu"


def test_unified_memory_takes_mps_for_both_workloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Apple Silicon there is no transfer to amortize, so MPS always wins."""
    _offer_mps(monkeypatch, unified_memory=True)
    assert resolve_device("auto", workload="interactive") == "mps"
    assert resolve_device("auto", workload="batch") == "mps"


def test_discrete_gpu_takes_mps_only_for_batch_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """The measured split: batches earn the transfer, single queries do not.

    Without this, every search on the reference machine would be ~3.7x slower
    than before accelerator support was added — a regression introduced in the
    name of acceleration, and invisible unless someone timed it.
    """
    _offer_mps(monkeypatch, unified_memory=False)
    assert resolve_device("auto", workload="batch") == "mps"
    assert resolve_device("auto", workload="interactive") == "cpu"


def test_interactive_is_the_default_workload(monkeypatch: pytest.MonkeyPatch) -> None:
    """The request path is the caller that must not forget to say which it is.

    Defaulting the other way would make an omission cost latency on exactly the
    path where latency is visible.
    """
    _offer_mps(monkeypatch, unified_memory=False)
    assert resolve_device("auto") == "cpu"
