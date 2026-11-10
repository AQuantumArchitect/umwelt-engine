# Glyph language (v0.4) — `umwelt.glyph`

An executable reading of a small language over a belief field: an emoji is a pole of a qubit, a ket is a statement, a bra is a question, and `›` descends into a pole's child scope. The package parses a sentence, binds it to a declared scope, and renders a `frame/1` document as sentences. Rendering reads; it never writes state.

**Grammar of record:** the page `docs/GLYPH-LANGUAGE.md` in the the-broadcast repository (grammar v0.1, with the v0.3 amendments of §15–§24 and the v0.4 amendments of §25–§28). This module implements it literally; where the page is silent the choices below are ours and are marked as decisions.

## Layout

- `normalize.py` — `canon`, `glyphs` (NFC; variation selectors stripped; ZWJ sequences whole; skin tones stripped unless the scope keeps them).
- `scope.py` — `Scope`, `Register`; `resolve` refuses `Ambiguous` / `Unbound`; `resolve_path` descends by pole. Scope version = hash of canonical content. A scope is declared separately from the engine spec (the spec's `AxisFace` has opaque pole labels but nothing a sentence can resolve against); `Register.node` optionally names the frame node it renders from.
- `num.py` — `Num`: a float with a provenance (measured / forecast / dark) and least-of-inputs arithmetic (v0.4).
- `parse.py` — the six forms and addresses to typed AST; `canonical` for round-trip.
- `bind.py` — ranges, population canonicalization to pole 0, evidence never symmetrized, dark kept distinct.
- `render.py` — `sentences_from_frame`, `frame_node`, `prose`.
- `receipt.py` — `Receipt`, `Measurement`, `Outcome`, `qq_check` (v0.3 §16), `grade` (v0.4 §26).
- `metrics.py` — `uhlmann_fidelity`, `bures_distance`, `holevo_chi`, `rho_from_bloch`, `rho_from_population`, `resonance` (〰), `tension` (⚡) (v0.3 §18).

## Decisions the page left open

1. Bare address `☀›🔥›⚫` (no form at the end) is `Address(path, None)`; all glyphs are path.
2. A bra's answer is written after `=`: `⟨☀|ψ⟩ = [.8]☀`. Allowed answers: ket, population, dark. `ψ` marks "the current state" (`Bra.psi`). The answer need not name the bra's register.
3. A glyph is any grapheme whose first code point is non-ASCII and not one of `⟨ ⟩ › ψ` (or a keycap sequence). ASCII letters and digits are never glyphs.
4. Numbers may be written `.8`, `0.8`, `1e-3`; canonical output is the shortest round-tripping `repr`, and the Bloch triple is written without spaces.
5. `[r,θ,φ]🌙☀` (pole 1 first) binds to the same fact as the pole-0-first spelling (θ→π−θ, φ→−φ).
6. Dark belongs to the register: `☀?` and `🌙?` bind to the same fact.
7. A ket may not name one register twice. φ is in [0,2π), so exactly 2π is refused.
8. Rendering: engine frames carry `bloch_z` values (a z-component), so `p = (1+z)/2` along pole 0; any other unit must already be in [0,1]. A node key `bloch` = [x,y,z] feeds `precise`. A stale node gets a trailing `# age Ns stale` comment; the parser ignores trailing comments.
9. A scope that keeps skin tones distinct requires `parse(..., keep_skin_tones=True)`.

## v0.3 rules (grammar §16–§19)

| Rule | Accepts | Refuses |
|---|---|---|
| Earned phase | `[r,θ,·]☀🌙` (dark phase) and `[r,θ,φ~]☀🌙` (model phase) bind with no receipt; `[r,θ,φ]` binds with a `Receipt` holding measurements (η > 0) of that register along two non-commuting axes | numeric φ with no receipt, an empty one, one axis measured twice, the register's two poles (☀ and 🌙 are one axis), another register's bases, or η = 0 entries. The parser accepts all three spellings; canonical printing keeps `·` and `~` |
| `qq_check(receipt)` | `✅` / `❌` for the QQ equality p(Ay,Bn)+p(An,By) = p(By,An)+p(Bn,Ay), given all four cells in both orders, within `tol` | returns `Dark` (`✅?`) when either order is missing; a duplicate cell is a `ReceiptError` |
| Earned tie | `\|☀⌢🔥⟩` parses; binds with a receipt entry `joint=True`, η > 0, naming every register of the ket | the tie without such an entry. Rendering emits `⌢` only for a frame node that carries a `joint_receipt` key (`{"with": node id, "poles": [i, j]}`) |
| Two descents | `☀›…` under a pole (`Register.children`); `☀🌙›…` under a register, a conditional state (`Scope.children`, keyed by register name). `resolve_path` returns `Step(scope_ref, register, pole, kind)`; `kind` is `"pole"` or `"register"` (pole is None for a register step) | a register step whose two glyphs are not one register's two poles; a descent where the scope declares no child of that kind |
| Resolution ε | `Scope(epsilon=…)`; a rendering from such a scope appends `# ε=…` | negative, non-finite or non-numeric ε. Nothing else changes |
| Resonance / tension | Uhlmann fidelity F = (tr √(√ρσ√ρ))², Bures d = √(2(1−√F)), Holevo χ in nats, on n×n states; `resonance`/`tension` take bound populations and Bloch statements | dark and evidence leaves (no state: dark is not a half). A Bloch node with a dark phase is read dephased |
| Product scope | `Scope.product(s1, s2)`: id `s1⊗s2`, registers are the pairs `a⊗b` (poles are glyph pairs); `\|☀₁🌙₂⟩` binds only there; `canonical_in(ast, scope)` writes the parent subscripts | the same ket in any other scope: `Unbound`, naming §19. Inside a product scope a glyph in both parents needs its subscript (`Ambiguous` otherwise); a ket takes exactly one glyph per parent |

### Decisions v0.3 left to us

1. Two measurements are non-commuting when their axes are neither parallel nor antiparallel. `Measurement.direction` is a pole glyph (the register's frame axis), `"x"`/`"y"`/`"z"`, or a 3-vector.
2. A tie must join every adjacent glyph of the ket or none; a tie over n registers needs one joint entry naming all n.
3. The renderer writes an engine Bloch vector's phase as `φ~` (the engine computes it from dynamics), and as `·` when r = 0.
4. A register step is stored as one path element of two glyphs (`"☀🌙"`); a bare address still ends on a single pole.
5. QQ takes its outcome probabilities from `Receipt.outcomes`, not from `Measurement`s.
6. Entropies are natural-log (nats); Holevo χ equals Jensen–Shannon in nats.
7. Parent subscripts are `₁`/`₂`; all subscript digits and `· ⌢` are reserved symbols, never pole glyphs.

## v0.4 — decision 14: every number is measured, forecast or dark (§25–§28)

14. **Universal `~`.** Every numeric slot (population, evidence η, Bloch r, θ, φ) is a `Num`: `0.8` **measured**, `0.8~` **forecast**, `?` **dark**. `φ~` is the special case; §23's "`~` on φ only" is superseded. `~~`, `~?`, `?~` and a bare `~` are refused.
    - **Earned.** `bind` refuses a straight number unless the receipt holds a measurement (η > 0) of that register, or the caller passes `engine_origin=True` (the engine adapter's declaration that the sentence was rendered from ρ built from receipts). The error names the slot and suggests `~`. A numeric φ still needs two bases; the engine flag never earns it.
    - **Composition is least-of-inputs** (measured < forecast < dark): one `~` input makes the result `~`, one `?` makes it `?`. `Num` arithmetic, `resonance`, `tension`, `holevo_chi` and `von_neumann_entropy` all return a `Num` that obeys it.
    - **Renderers never drop a `~`**: glyph string `~`; prose "about N (forecast)" (straight: the bare number; dark: "unknown (dark)"); `frame_node` / `sentences_from_frame` carry `"forecast": true` beside the value (a list of three bools beside `bloch_rtp` for a vector). `tests/test_glyph_roundtrip.py` fails if any of them does.
    - **Grading.** `grade(forecast, measured, tol=)` writes `🧪 ⟨measured|forecast⟩ → |✅⟩` (or `|❌⟩`, or `✅?` if the measurement is dark) and returns a tape record.
    - Decisions the page left: a dark p, r or θ binds to `BDark`; a dark η is refused; a dephased reading (dark φ) does not darken a metric; `?` is accepted in the φ slot and canonicalises to `·`; costs, dates, counts and durations have no slot in this module yet (`parse_number` reads the literal).

## Not implemented

- Verbs (v0.2 §9–§12: the Strand Dance operators and lexemes). `🔮` is not a verb here; the two descents are only the address forms.
- QQ beyond two binary questions.
- Other forms on a product scope (populations and the bridging-intercept / factionality statements of §19); only the cross-scope ket and the `⚡` between the parents' states.
- `ε` is declared and stated; no distortion is computed from it.
- The language-paper chain study (§21).

## Limits

- The grapheme splitter is a small emoji-aware splitter, not UAX #29: flags, keycaps, tag sequences, combining marks and ZWJ chains only. Other scripts' clustering is not modelled.
- `|☀🔥⟩` is siblings read jointly, nothing more; a non-separability claim is the earned tie only.
- No verbs (Strand Dance operations are not in the language yet).
- Two-pole registers only; no qutrits. No time or accumulated-phase form.
- A glyph shared by two registers cannot be addressed at all in v0.1 (there is no register-id syntax); the scope must be fixed.
- The grammar page is a draft and may change under this module.
