from umwelt.glyph import canon, glyphs

SUN, SUN_EMOJI = "☀", "☀️"
FAMILY = "\U0001F468‍\U0001F469‍\U0001F467"


def test_variation_selector_is_one_glyph():
    assert canon(SUN) == canon(SUN_EMOJI) == SUN
    assert canon("☀︎") == SUN


def test_zwj_family_is_one_glyph():
    assert glyphs(FAMILY) == [FAMILY]
    assert len(glyphs(f"{FAMILY}☀")) == 2


def test_skin_tones_stripped_unless_kept():
    wave = "\U0001F44B"
    toned = wave + "\U0001F3FD"
    assert canon(toned) == wave
    assert canon(toned, keep_skin_tones=True) == toned
    assert glyphs(toned, keep_skin_tones=True) == [toned]
    assert canon("\U0001F468\U0001F3FD‍\U0001F469\U0001F3FD") == "\U0001F468‍\U0001F469"


def test_flags_and_keycaps_and_round_trip():
    us, de = "\U0001F1FA\U0001F1F8", "\U0001F1E9\U0001F1EA"
    assert glyphs(us + de) == [us, de]
    assert glyphs("1⃣") == ["1⃣"]
    s = f"a|{FAMILY}{SUN_EMOJI}{us} ́"
    assert "".join(glyphs(s)) == canon(s)


def test_nfc_and_idempotence():
    assert canon("é") == "é"
    s = canon(f"{SUN_EMOJI}\U0001F44B\U0001F3FD")
    assert canon(s) == s
