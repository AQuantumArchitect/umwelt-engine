"""The field DAG + git leg, finally under test (this module had none).

1. EMPTY-DIFF CERTIFICATE at the DAG level: an unchanged field writes no node and
   appends nothing; a real coupling change writes exactly one parent-linked node.
2. README #4 as a test: against a git-init'ed archive, one training event = one
   commit; a frozen re-record leaves `git log` AND the worktree diff empty.
3. The Poincaré strobe: with a clock cluster, nodes are recorded only in the rest
   window; with no clock, unconditionally (a world without a rhythm has no section).
4. Failure honesty: a non-git archive still writes the node, no exception.
5. The growth service's field-shaped caller works (it passes a bare
   SimpleNamespace(clusters=...), which used to die on `.field` inside the ledger's
   own try/except — the production learning event was silently never recorded).
"""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from umwelt.substrate.cumulant_cluster import CumulantCluster
from umwelt.substrate.field_gauge import (
    field_lineage, latest_field_node, maybe_record_field_node, record_field_node)


def _cluster(j=0.3):
    c = CumulantCluster("m", ["a", "b"], gamma=0.1, dt=0.01,
                        role_modes={"a": "unitary", "b": "unitary"})
    c.e1[0] = (0.2, 0.1, 0.4)
    c.e1[1] = (0.0, -0.3, 0.6)
    c.set_couplings(h_fields=np.array([[0.0, 0.0, 0.5], [0.3, 0.0, 0.0]]), zz={(0, 1): j})
    return c


def _reservoir(j=0.3, with_clock_phase=None):
    clusters = {"m": _cluster(j)}
    if with_clock_phase is not None:
        clk = CumulantCluster("_clock", ["phase"], gamma=0.0, dt=0.01,
                              role_modes={"phase": "unitary"})
        phi = 2.0 * np.pi * with_clock_phase
        clk.e1[0] = (np.cos(phi), np.sin(phi), 0.0)
        clusters["_clock"] = clk
    return SimpleNamespace(field=SimpleNamespace(clusters=clusters, graph=None), _step=7)


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args],
                          capture_output=True, text=True)


class TestEmptyDiffCertificate:
    def test_unchanged_field_writes_nothing(self, tmp_path):
        r = _reservoir()
        first = record_field_node(r, archive_root=tmp_path, note="boot")
        assert first is not None
        index_bytes = (tmp_path / "index.jsonl").read_bytes()
        again = record_field_node(r, archive_root=tmp_path, note="idle")
        assert again is None                                  # the empty diff
        assert (tmp_path / "index.jsonl").read_bytes() == index_bytes

    def test_training_event_writes_one_parent_linked_node(self, tmp_path):
        record_field_node(_reservoir(j=0.3), archive_root=tmp_path)
        prev = latest_field_node(tmp_path)
        node = record_field_node(_reservoir(j=0.45), archive_root=tmp_path, note="learned")
        assert node is not None
        assert node["parents"] == [prev["sha256"][:8]]
        assert len(field_lineage(tmp_path)) == 2
        # both hashes + the invariant sheet ride every node
        assert node["params"]["canon_hash"] != prev["params"]["canon_hash"]
        assert node["params"]["canon_hash_v2"] != prev["params"]["canon_hash_v2"]
        assert node["invariants"]["per_cluster"]["m"] != \
            prev["invariants"]["per_cluster"]["m"]


class TestGitLeg:
    def _init_archive(self, root):
        _git(root, "init", "-q")
        _git(root, "config", "user.email", "gauge@test")
        _git(root, "config", "user.name", "gauge test")

    def test_one_training_event_one_commit_frozen_leaves_git_clean(self, tmp_path):
        self._init_archive(tmp_path)
        node = record_field_node(_reservoir(j=0.3), archive_root=tmp_path,
                                 note="boot", git_commit=True)
        assert node is not None
        log1 = _git(tmp_path, "log", "--oneline").stdout.strip().splitlines()
        assert len(log1) == 1                                  # one event, one commit
        # the frozen re-record: no node, no commit, clean worktree — README #4 as a test
        assert record_field_node(_reservoir(j=0.3), archive_root=tmp_path,
                                 git_commit=True) is None
        log2 = _git(tmp_path, "log", "--oneline").stdout.strip().splitlines()
        assert log2 == log1
        assert _git(tmp_path, "status", "--porcelain").stdout.strip() == ""
        # a training event commits exactly one more
        assert record_field_node(_reservoir(j=0.5), archive_root=tmp_path,
                                 git_commit=True) is not None
        log3 = _git(tmp_path, "log", "--oneline").stdout.strip().splitlines()
        assert len(log3) == 2
        assert _git(tmp_path, "status", "--porcelain").stdout.strip() == ""

    def test_non_git_archive_writes_node_without_exception(self, tmp_path):
        node = record_field_node(_reservoir(), archive_root=tmp_path, git_commit=True)
        assert node is not None
        assert (tmp_path / node["stored"]).exists()

    def test_env_flag_default_off(self, tmp_path, monkeypatch):
        monkeypatch.delenv("UMWELT_FIELD_GIT", raising=False)
        self._init_archive(tmp_path)
        node = maybe_record_field_node(_reservoir(), archive_root=tmp_path)
        assert node is not None
        assert _git(tmp_path, "log", "--oneline").returncode != 0 or \
            _git(tmp_path, "log", "--oneline").stdout.strip() == ""   # no commit made
        # opting in via the env flag commits
        monkeypatch.setenv("UMWELT_FIELD_GIT", "1")
        assert maybe_record_field_node(_reservoir(j=0.9), archive_root=tmp_path) is not None
        assert len(_git(tmp_path, "log", "--oneline").stdout.strip().splitlines()) == 1


class TestPoincareStrobe:
    def test_mid_cycle_defers_rest_window_records(self, tmp_path):
        busy = _reservoir(with_clock_phase=0.9)               # outside (0.45, 0.65)
        assert maybe_record_field_node(busy, archive_root=tmp_path) is None
        assert not (tmp_path / "index.jsonl").exists()
        resting = _reservoir(with_clock_phase=0.5)            # in the rest window
        assert maybe_record_field_node(resting, archive_root=tmp_path) is not None

    def test_no_clock_records_unconditionally(self, tmp_path):
        assert maybe_record_field_node(_reservoir(), archive_root=tmp_path) is not None

    def test_strobe_off_bypasses_the_section(self, tmp_path):
        busy = _reservoir(with_clock_phase=0.9)
        assert maybe_record_field_node(busy, archive_root=tmp_path, strobe=False) is not None


class TestFieldShapedCaller:
    def test_growth_services_bare_field_works(self, tmp_path):
        """The production caller passes SimpleNamespace(clusters=..., _step=0) — not a
        reservoir. This used to raise AttributeError('.field') inside the caller's own
        try/except, so the 'provable learning event' was never actually recorded."""
        bare = SimpleNamespace(clusters={"m": _cluster()}, _step=0)
        node = maybe_record_field_node(bare, archive_root=tmp_path, note="field_growth m")
        assert node is not None
        rec = json.loads((tmp_path / "index.jsonl").read_text().splitlines()[0])
        assert rec["note"] == "field_growth m"
        assert rec["params"]["n_qubits"] == 2
