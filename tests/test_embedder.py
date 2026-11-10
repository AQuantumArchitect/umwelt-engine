"""Symbol codebook — grounding names cells; confident unlabeled visits stay latent."""
from __future__ import annotations

from umwelt.learning.embedder import SymbolCodebook


def test_ground_and_de_embed_known_pole():
    cb = SymbolCodebook(n_cells=32)
    cb.ground_axis({"+z": "high", "-z": "low"})
    hi = cb.de_embed((0.0, 0.0, 0.95), purity=0.9)
    lo = cb.de_embed((0.0, 0.0, -0.95), purity=0.9)
    assert hi.label == "high" and lo.label == "low"
    assert hi.minted is False and lo.minted is False
    assert cb.n_minted() == 0


def test_confident_unlabeled_is_latent_not_named():
    cb = SymbolCodebook(n_cells=32)
    sym = cb.de_embed((0.0, 0.0, 0.95), purity=0.95, mint=True)
    assert sym.label is None
    assert sym.latent is True
    assert sym.minted is False
    assert "concept_" not in str(cb.vocabulary())
    assert cb.n_latent() == 1
    assert cb.latent_events[-1]["label"] is None
    # app names it
    cb.ground("named", (0.0, 0.0, 0.95))
    named = cb.de_embed((0.0, 0.0, 0.95), purity=0.95)
    assert named.label == "named"
    assert named.latent is False


def test_embed_inverts_ground():
    cb = SymbolCodebook(n_cells=32)
    cb.ground("high", (0.0, 0.0, 1.0))
    anchor = cb.embed("high")
    assert anchor is not None
    assert abs(anchor[2] - 1.0) < 0.25          # nearest cell to +z
    assert cb.embed("never-grounded") is None
