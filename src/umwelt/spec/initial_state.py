"""Explicit independent local boot priors; preparation is not outside evidence."""
from collections.abc import Mapping
import math


def initial_bloch_errors(spec) -> list[str]:
    errors = []
    for node in spec.nodes:
        prior = getattr(node, 'initial_bloch', None)
        if prior is None:
            continue  # historical contract: legacy seed profile and role registry
        where = f'node {node.name!r} initial_bloch'
        if getattr(node, 'folded', False):
            errors.append(f'{where}: a folded node has no independent cluster to prepare')
        if not isinstance(prior, Mapping):
            errors.append(f'{where}: expected a complete role -> Bloch-vector mapping or None')
            continue
        if set(prior) != set(node.roles):
            errors.append(f'{where}: must name exactly all declared roles {tuple(node.roles)!r}')
        for role, vec in prior.items():
            if (not isinstance(vec, (tuple, list)) or len(vec) != 3
                    or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                               and -1 <= v <= 1 for v in vec)):
                errors.append(f'{where}[{role!r}]: expected three finite real coordinates in [-1, 1]')
            elif math.hypot(*vec) > 1:
                errors.append(f'{where}[{role!r}]: Bloch-vector norm must be <= 1')
    return errors


def initial_bloch_priors(spec) -> dict:
    """Validate before construction and detach vectors from caller-owned containers."""
    errors = initial_bloch_errors(spec)
    if errors:
        raise ValueError('; '.join(errors))
    return {n.name: {r: tuple(float(v) for v in vec) for r, vec in n.initial_bloch.items()}
            for n in spec.nodes if getattr(n, 'initial_bloch', None) is not None}


def apply_initial_bloch(engine, priors: dict, *, scoped: bool = False) -> None:
    """Apply after legacy profiles, before forecasting. Scoped-out clusters stay absent.

    observe_qubit is the substrate's state preparation operation; unlike measure_axis,
    it creates no measurement receipt, bridge timestamp or earned signal. These local
    states specify a product preparation, not a correlated/entangled joint prior.
    """
    for name, values in priors.items():
        cluster = engine.field.clusters.get(name)
        if cluster is None and values and not scoped:
            raise ValueError(f'node {name!r} initial_bloch: no cluster exists for preparation')
        if cluster is not None and not set(values) <= set(cluster.role_index):
            raise ValueError(f'node {name!r} initial_bloch: built cluster has different roles')
    for name, values in priors.items():
        cluster = engine.field.clusters.get(name)
        if cluster is None:
            continue  # cluster_filter intentionally omitted this node
        for role, vec in values.items():
            cluster.observe_qubit(cluster.role_index[role], vec, alpha=1.)
