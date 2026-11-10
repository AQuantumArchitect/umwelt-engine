# Engine and application test boundaries

Applications depend on the engine; the engine does not depend on them. This repository ships
no application package. A private downstream application used to live under
the `umwelt.consumers` namespace; its source, tests and docs moved to its own private
repository on 2026-11-08 (layering law, `docs/VISION.md` II).

What this repository enforces:

- `tests/test_vocabulary_lint.py` bans domain vocabulary in all of `src/umwelt/`, with no exemption.
- `tests/test_consumer_boundary.py` fails if any module under `src/` imports `umwelt.consumers`
  or the downstream application's package. It is kept as the regression guard against an application re-entering the engine.

CI runs the engine tests and proofs on Python 3.10 and 3.12 with `pip install -e .[dev]` and no
private credentials.

A consumer tests itself, in its own repository, against a pinned engine. See
[CONSUMER_PIN.md](CONSUMER_PIN.md) for the pin shape. Public engine surface is described in
[STABILITY.md](STABILITY.md); a consumer that imports outside it should say so and expect to
follow engine changes by hand.
