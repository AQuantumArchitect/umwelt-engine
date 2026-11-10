"""Shared test scope: the illustrative Terrarium household poles of grammar §7."""
from umwelt.glyph import Measurement, Receipt, Register, Scope

# Child scopes opened by poles, so the §7 addresses descend. Poles are illustrative.
_shelter_child = Scope("terrarium/shelter", {
    "repair": Register("🔧", "🪛", children={
        0: Scope("terrarium/shelter/repair", {"need": Register("🩹", "✨")})}),
    "energy": Register("🔋", "🔌", children={
        0: Scope("terrarium/shelter/energy", {"charge": Register("🪫", "⚡")})}),
})


def terrarium() -> Scope:
    return Scope("terrarium/household", {
        "water": Register("💧", "🏜", node="water"),
        "food": Register("🥕", "🍽"),
        "shelter": Register("🏠", "⛺", children={0: _shelter_child}),
        "energy": Register("🔋", "🪫"),
        "repair": Register("🔧", "🩹"),
        "support": Register("🤝", "🚪"),
    })


SECTION_7 = [
    "💧?",
    "🥕!0.4",
    "[.7]🔋",
    "[0.9, 0.3, 1.2]🔋🪫",
    "|🏠🤝⟩",
    "⟨🤝|",
    "🏠›🔧›🩹!0.92",
    "🏠›🔋›🪫?",
]


def two_basis_receipt(*registers: str, eta: float = 0.9) -> Receipt:
    """A receipt measuring each named register along its frame axis and along x."""
    ms = []
    for i, r in enumerate(registers):
        ms += [Measurement(r, (0, 0, 1), eta, 0), Measurement(r, "x", eta, 1)]
    return Receipt(tuple(ms))


def ebind(ast, scope, receipt=None, **kw):
    """``bind`` as the engine adapter calls it: straight numbers count as measured (v0.4 §25).
    For tests about other behaviours; the earn rule itself is tested with the real ``bind``."""
    from umwelt.glyph import bind
    kw.setdefault("engine_origin", True)
    return bind(ast, scope, receipt, **kw)
