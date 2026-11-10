# Installing and pinning umwelt-engine

How an application depends on the engine. It is not on PyPI; you install it from the
public git repository, pinned to something immutable.

## Names

| What | Name |
|---|---|
| Repository | `https://github.com/AQuantumArchitect/umwelt-engine` |
| Distribution (what you `pip install`) | `umwelt-engine` |
| Import package | `umwelt` (plus `umweltd`, `umweltforge`) |

**Import-name collision.** An unrelated PyPI project named `umwelt` (a small
pydantic-env config package) also installs an import package called `umwelt`. Never
install both into one environment: whichever is installed last wins the import, and
the failure is confusing. `pip install umwelt` is *not* this engine. Always install
`umwelt-engine`, and from git as below. If `import umwelt` gives you something without
`DomainSpec`, check `pip list` for the other distribution.

## Version policy

One place: [STABILITY.md](STABILITY.md) (SemVer; while 0.x, a minor bump may remove a
deprecated public name, a patch never breaks one). Releases are git tags `vX.Y.Z`;
a published tag never moves. Release notes are in [CHANGELOG.md](../CHANGELOG.md).
While 0.x, pin exactly.

## Install

Pin a release tag, or a full 40-character commit SHA. Never a branch or short SHA.

```bash
pip install "umwelt-engine @ git+https://github.com/AQuantumArchitect/umwelt-engine@v0.2.1"
```

`requirements.txt`:

```
umwelt-engine @ git+https://github.com/AQuantumArchitect/umwelt-engine@v0.2.1
```

`pyproject.toml` of your app:

```toml
dependencies = [
    "umwelt-engine @ git+https://github.com/AQuantumArchitect/umwelt-engine@v0.2.1",
]
# or, an exact commit:
#   "umwelt-engine @ git+https://github.com/AQuantumArchitect/umwelt-engine@<full-40-char-sha>"
```

Extras work as usual: `umwelt-engine[rollout] @ git+https://…@v0.2.1`.
No credentials are needed; do not put any in a dependency URL.

The current release is `v0.2.1`. Replace the tag with a later one when you upgrade.

### Record the pin

A pin without a "verified by" command is a wish. In your repo keep a row such as:

```
| umwelt-engine | `v0.2.1` | verified by: python -m pytest -q → <N> passed |
```

### Dev option: editable sibling checkout

On your own machine only, never in `pyproject.toml` or CI:

```bash
pip install -e ../umwelt-engine
```

## Upgrade

1. Read [CHANGELOG.md](../CHANGELOG.md) between your tag and the new one; behaviour
   changes that move numbers or hashes are listed there.
2. Change the tag (or SHA) in one commit.
3. `pip install --upgrade --force-reinstall --no-deps umwelt-engine @ git+…@<new>`
   (or reinstall your requirements) and run your own gates.
4. Update your pin record with the new gate output.

If you replay against recorded hashes, expect to re-record on any bump that the
changelog flags as a behaviour change.

## Import from the facade

Import from `umwelt` (`from umwelt import DomainSpec, GameHost`), not from deep paths.
The top-level names are the stable surface ([API.md](API.md)); deep paths are internal
([STABILITY.md](STABILITY.md)).

## Don'ts

- **Don't** use `file:///home/...` URLs; they break every machine but yours.
- **Don't** float on `main`; replay/certificate consumers break silently when
  snapshot shapes drift.
- **Don't** vendor engine source into your tree; you will fork the integrator
  without noticing.
