"""Select a bounded monomial basis without materializing discarded combinations."""
import itertools
import math

import numpy as np


def _ranks(rng, population, count):
    if population <= np.iinfo(np.int64).max:
        # Keep the historical NumPy draw exactly: changing a feature's identity
        # would invalidate the learned coefficient attached to that coordinate.
        return sorted(int(i) for i in rng.choice(population, size=count, replace=False))
    # Floyd sampling for populations beyond NumPy's signed-integer range.
    # Memory is O(count), even when the population has hundreds of digits.
    selected = set()
    for upper in range(population - count + 1, population + 1):
        bits = (upper - 1).bit_length()
        mask = (1 << bits) - 1
        while True:
            choice = int.from_bytes(rng.bytes((bits + 7) // 8), "little") & mask
            if choice < upper:
                break
        selected.add(upper - 1 if choice in selected else choice)
    return sorted(selected)


def _combination(n, degree, rank):
    """Decode the lexicographic rank used by itertools.combinations."""
    chosen = []
    start = 0
    for remaining in range(degree - 1, -1, -1):
        for atom in range(start, n - remaining):
            block = math.comb(n - atom - 1, remaining)
            if rank < block:
                chosen.append(atom)
                start = atom + 1
                break
            rank -= block
    return tuple(chosen)


def select_monomials(n_atoms, order, cap):
    """Historical deterministic basis, with allocation proportional to the cap.

    cap=0 retains the explicit uncapped contract. A positive cap bounds selected
    features; combinatorial population sizes are represented as integer counts.
    """
    if cap < 0:
        raise ValueError("max_context_features must be nonnegative")
    degrees = range(1, min(order, n_atoms) + 1)
    counts = [math.comb(n_atoms, degree) for degree in degrees]
    total = sum(counts)
    if cap == 0 or total <= cap:
        return tuple(row for degree in degrees
                     for row in itertools.combinations(range(n_atoms), degree))
    rng = np.random.default_rng(0)
    if cap <= n_atoms:
        ranks = _ranks(rng, total, cap)
        selected = []
    else:
        ranks = [rank + n_atoms for rank in _ranks(rng, total - n_atoms, cap - n_atoms)]
        selected = [(atom,) for atom in range(n_atoms)]
    for rank in ranks:
        for degree, count in zip(degrees, counts):
            if rank < count:
                selected.append(_combination(n_atoms, degree, rank))
                break
            rank -= count
    return tuple(selected)
