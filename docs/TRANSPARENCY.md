# Transparency — how a world confesses

The field is the mind. These projections are **read-only membranes**: they never
act, never invent a label from geometry, and they render missing evidence as
absence.

| Face | What you see | Honesty caveat |
|---|---|---|
| **Nameplate** (`projection.nameplates`) | One row per `node.role`: pole labels, status, Bloch z, purity | Status is evidence (`latent` never observed / `live` / `dark` last η=0). No ρ, no tick copied in. |
| **Confession** (`projection.confession`) | sensed → believed → forecast → decided → dispatched? → echo | Forecast **used** only if skill-vs-persistence > 0. Shadow decisions stay `dispatched=false`. Echo is null unless independently tagged. |
| **Glass door** (`projection.glass_door`) | ≤5 tiles | Drill is a nameplate link, not a control room. Empty atlas → `status=dark`. `null_dispatch` is always true on this face. |
| **Cognifold** (`projection.cognifold_trace`) | Bloch spherical chart + couplings + forecast ladder | Radius **is** confidence. Steel edges = Hamiltonian skeleton, not live correlation. Mint lines (`mi`) are the pulse. |

The playground (`/ui`) shows the glass door and one confession sentence first; the
lab (belief bars, raw `graph_state`) stays below.

Naming a cell is an **app write** (`learning.embedder.SymbolCodebook.ground`).
Confident unlabeled visits are `latent` with `label=null`. The engine never mints
`concept_N`.

Daemon routes: `GET /nameplates`, `GET /confession`, `GET /glass_door`,
`GET /cognifold`. See [SERVICE.md](SERVICE.md).
