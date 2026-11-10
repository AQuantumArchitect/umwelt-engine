"""examples/n_gym_body — DomainSpec leaf for N-gym body capacity beliefs.

Grows the umwelt model so *enough free muscle* is a field over nodes, not a
hardcoded BAR constant or a one-off coral organ.

Topology (data only):
  n_gym (root)
    ├── body          free capacity beliefs (cpu/mem/gpu free, busy_load)
    ├── portfolio     workstream rank / pack / exclusive locks
    └── stream.*      one region per workstream class (heavy vs light)

Bindings are report-shaped (force_observe): organic rest/work claims land as
observations that hold between sessions — same lesson as hive_relay.

Not a product gym: BAR and SpaceWheat are external actuators that *feed* this
field; the field grades skill, not win-rate.
"""
from __future__ import annotations

from umwelt.spec.schema import (
    BindingSpec,
    BridgeSpec,
    DomainSpec,
    NodeSpec,
    OutputSpec,
)


def n_gym_body_spec() -> DomainSpec:
    """Body + portfolio + two stream classes as a DomainSpec."""
    nodes = [
        NodeSpec(
            "n_gym",
            parent=None,
            kind="root",
            roles=("skill", "dark"),
            role_modes={"skill": "dissipative", "dark": "unitary"},
            params={"gamma_diss": (0.02, 0.01, 0.001, 0.15)},
        ),
        NodeSpec(
            "body",
            parent="n_gym",
            kind="environment",
            roles=("free_cpu", "free_mem", "busy_load", "enough"),
            role_modes={
                "free_cpu": "dissipative",
                "free_mem": "dissipative",
                "busy_load": "dissipative",
                "enough": "unitary",
            },
            # gamma_diss so free floats can settle between probes without sticky pole
            params={
                "gamma_diss_free_cpu": (0.05, 0.02, 0.001, 0.3),
                "gamma_diss_free_mem": (0.05, 0.02, 0.001, 0.3),
                "gamma_diss_busy_load": (0.08, 0.02, 0.001, 0.4),
            },
            features=("capacity", "muscle"),
        ),
        NodeSpec(
            "portfolio",
            parent="n_gym",
            kind="component",
            roles=("rank", "lease", "pack"),
            role_modes={"rank": "unitary", "lease": "unitary", "pack": "dissipative"},
            features=("scheduler",),
        ),
        NodeSpec(
            "stream_heavy",
            parent="n_gym",
            kind="region",
            roles=("debt", "starvation", "hose_ok"),
            role_modes={
                "debt": "dissipative",
                "starvation": "dissipative",
                "hose_ok": "unitary",
            },
            features=("capacity_class_heavy", "exclusive_heavy_slot"),
            params={"gamma_diss": (0.03, 0.01, 0.001, 0.2)},
        ),
        NodeSpec(
            "stream_light",
            parent="n_gym",
            kind="region",
            roles=("debt", "starvation", "hose_ok"),
            role_modes={
                "debt": "dissipative",
                "starvation": "dissipative",
                "hose_ok": "unitary",
            },
            features=("capacity_class_light",),
            params={"gamma_diss": (0.04, 0.01, 0.001, 0.25)},
        ),
    ]
    bridges = (
        BridgeSpec("body", "portfolio", shared_roles=(), kind="open"),
        BridgeSpec("portfolio", "stream_heavy", shared_roles=("debt",), kind="gated"),
        BridgeSpec("portfolio", "stream_light", shared_roles=("debt",), kind="open"),
        BridgeSpec(
            "body",
            "stream_heavy",
            shared_roles=(),
            kind="tendril",
            role_map={"free_cpu": "hose_ok"},
        ),
    )
    # Report-shaped measurements: organic cycle posts free/busy/rest/work claims
    bindings = (
        BindingSpec(
            "probe_free_cpu",
            zone="body",
            role="free_cpu",
            normalizer={"type": "range", "lo": 0.0, "hi": 1.0},
            force_observe=True,
            collapse_alpha=0.35,
            description="Measured free CPU fraction 0..1",
        ),
        BindingSpec(
            "probe_free_mem",
            zone="body",
            role="free_mem",
            normalizer={"type": "range", "lo": 0.0, "hi": 1.0},
            force_observe=True,
            collapse_alpha=0.35,
        ),
        BindingSpec(
            "probe_busy_load",
            zone="body",
            role="busy_load",
            normalizer={"type": "range", "lo": 0.0, "hi": 1.0},
            force_observe=True,
            collapse_alpha=0.4,
        ),
        BindingSpec(
            "body_enough_ok",
            zone="body",
            role="enough",
            normalizer="binary",
            force_observe=True,
            collapse_alpha=0.5,
            description="1 if free met capacity.* enough for heavy act",
        ),
        BindingSpec(
            "heavy_hose_ok",
            zone="stream_heavy",
            role="hose_ok",
            normalizer="binary",
            force_observe=True,
            collapse_alpha=0.45,
        ),
        BindingSpec(
            "heavy_skill_debt",
            zone="stream_heavy",
            role="debt",
            normalizer={"type": "range", "lo": 0.0, "hi": 1.0},
            force_observe=True,
            collapse_alpha=0.3,
        ),
        BindingSpec(
            "light_hose_ok",
            zone="stream_light",
            role="hose_ok",
            normalizer="binary",
            force_observe=True,
            collapse_alpha=0.4,
        ),
        BindingSpec(
            "portfolio_lease_held",
            zone="portfolio",
            role="lease",
            normalizer="binary",
            force_observe=True,
            collapse_alpha=0.5,
        ),
    )
    outputs = (
        OutputSpec(
            "shadow_commit_muscle",
            node="portfolio",
            role="rank",
            kind="scalar",
            shadow=True,
        ),
        OutputSpec(
            "shadow_body_enough",
            node="body",
            role="enough",
            kind="binary",
            shadow=True,
        ),
    )
    return DomainSpec(
        name="n-gym-body",
        nodes=tuple(nodes),
        bridges=bridges,
        bindings=bindings,
        outputs=outputs,
        drivers=(),
        # sparse organic claims — hold between sessions (hive_relay lesson)
        ingest_hold_s=60.0,
        extensions={
            "thesis": (
                "N-gym body + portfolio field. Capacity enough is belief on body, "
                "not a product CPU organ. Gyms feed sensors; skill is primary."
            ),
            "coral_faces": {
                "capacity": "commons/vitals/organism_body_params.json#groups.capacity",
                "portfolio": "commons/vitals/muscle_portfolio.json",
                "registry": "commons/vitals/workstream_registry.json",
            },
        },
    )


def synthetic_capacity_rows(
    *,
    free_cpu: float = 0.45,
    free_mem: float = 0.22,
    busy_load: float = 0.55,
    heavy_ok: bool = True,
) -> list[tuple[str, str, str, None]]:
    """Minimal report stream for blank-slate / demo (ts, sensor_id, value, None)."""
    ts = "2026-08-03T12:00:00+00:00"
    enough = "1" if (free_cpu >= 0.42 and free_mem >= 0.18) else "0"
    return [
        (ts, "probe_free_cpu", f"{free_cpu:.4f}", None),
        (ts, "probe_free_mem", f"{free_mem:.4f}", None),
        (ts, "probe_busy_load", f"{busy_load:.4f}", None),
        (ts, "body_enough_ok", enough, None),
        (ts, "heavy_hose_ok", "1" if heavy_ok else "0", None),
        (ts, "heavy_skill_debt", "0.3" if heavy_ok else "0.8", None),
        (ts, "light_hose_ok", "1", None),
        (ts, "portfolio_lease_held", "0", None),
    ]


if __name__ == "__main__":
    spec = n_gym_body_spec()
    print(f"domain={spec.name} nodes={len(spec.nodes)} bindings={len(spec.bindings)}")
    for row in synthetic_capacity_rows()[:4]:
        print(row)
