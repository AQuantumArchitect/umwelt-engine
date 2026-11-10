import random

from glyph_fixtures import ebind, two_basis_receipt
from umwelt.glyph import Register, Scope, canonical, parse

POOL = list("☀🌙🔥⚫💧🏜🥕🍽🏠⛺🔋🪫🔧🩹🤝🚪🌲🌊🐟🦊🍄🌵⭐🧊🪨🛖") + ["👨‍👩‍👧", "🇺🇸"]


def rand_scope(rng, _unused, depth=0, sid="s"):
    glyphs = POOL[:]                      # each scope draws its own distinct poles
    rng.shuffle(glyphs)
    regs = {}
    n = rng.randint(2, 4)
    for k in range(n):
        p0, p1 = glyphs.pop(), glyphs.pop()
        kids = {}
        if depth < 2:
            for pole in (0, 1):
                if rng.random() < 0.5:
                    kids[pole] = rand_scope(rng, None, depth + 1, f"{sid}/{k}{pole}")
        regs[f"r{k}"] = Register(p0, p1, children=kids)
    return Scope(sid, regs)


def num(rng, lo, hi, open_lo=False):
    x = rng.uniform(lo, hi)
    if open_lo and x == 0:
        x = hi
    style = rng.choice(["{:.3f}", "{:.6g}", "{!r}", "{:e}"])
    t = style.format(x) if "{!r}" not in style else repr(x)
    if t.startswith("0.") and rng.random() < 0.5:
        t = t[1:]
    return t


def rand_leaf(rng, sc):
    regs = list(sc.registers.values())
    r = rng.choice(regs)
    g = rng.choice([r.pole0, r.pole1])
    sp = rng.choice(["", " "])
    form = rng.choice(["ket", "pop", "ev", "bloch", "dark", "bra", "psi", "ans"])
    if form == "ket":
        ks = rng.sample(regs, rng.randint(1, len(regs)))
        return "|" + sp.join(rng.choice([k.pole0, k.pole1]) for k in ks) + "⟩"
    if form == "pop":
        return f"[{sp}{num(rng, 0, 1)}{sp}]{g}"
    if form == "ev":
        return f"{g}!{num(rng, 0, 1, True)}"
    if form == "bloch":
        a, b = (r.pole0, r.pole1) if rng.random() < .5 else (r.pole1, r.pole0)
        return f"[{num(rng, 0, 1)},{sp}{num(rng, 0, 3.14)},{sp}{num(rng, 0, 6.28)}]{a}{sp}{b}"
    if form == "dark":
        return f"{g}?"
    if form == "bra":
        return f"⟨{g}|"
    if form == "psi":
        return f"⟨{g}|ψ⟩"
    return f"⟨{g}|ψ⟩ = [{num(rng, 0, 1)}]{g}"


def rand_sentence(rng, sc):
    path, cur = [], sc
    while rng.random() < 0.6:
        opts = [(r, p) for r in cur.registers.values() for p in r.children]
        if not opts:
            break
        r, p = rng.choice(opts)
        path.append(r.glyph(p))
        cur = r.children[p]
    if path and rng.random() < 0.15:                       # bare address
        r = rng.choice(list(cur.registers.values()))
        return "›".join(path + [rng.choice([r.pole0, r.pole1])])
    return "›".join(path + [rand_leaf(rng, cur)])


def test_parse_canonical_identity_300_cases():
    rng = random.Random(20261108)
    for i in range(300):
        sc = rand_scope(rng, None)
        text = rand_sentence(rng, sc)
        a = parse(text)
        c = canonical(a)
        assert parse(c) == a, (text, c)
        assert canonical(parse(c)) == c
        rc = two_basis_receipt(*(f"r{k}" for k in range(4)))
        assert ebind(a, sc, rc) == ebind(parse(c), sc, rc), text


def test_v03_forms_round_trip_300_cases():
    rng = random.Random(30303)
    for i in range(300):
        sc = rand_scope(rng, None)
        regs = list(sc.registers.values())
        r = rng.choice(regs)
        kind = rng.choice(["dark", "model", "tie", "sub", "regdesc"])
        if kind == "dark":
            text = f"[{num(rng, 0, 1)},{num(rng, 0, 3.14)},·]{r.pole0}{r.pole1}"
        elif kind == "model":
            text = f"[{num(rng, 0, 1)},{num(rng, 0, 3.14)},{num(rng, 0, 6.28)}~]{r.pole0}{r.pole1}"
        elif kind == "tie":
            ks = rng.sample(regs, 2)
            text = "|" + "⌢".join(rng.choice([k.pole0, k.pole1]) for k in ks) + "⟩"
        elif kind == "sub":
            text = f"|{r.pole0}{rng.choice('₁₂')}{r.pole1}{rng.choice('₁₂')}⟩"
        else:
            text = f"{r.pole0}{r.pole1}›{rng.choice([r.pole0, r.pole1])}?"
        a = parse(text)
        assert parse(canonical(a)) == a and canonical(parse(canonical(a))) == canonical(a), text
        if kind in ("dark", "model"):
            assert ebind(a, sc) == ebind(parse(canonical(a)), sc)
