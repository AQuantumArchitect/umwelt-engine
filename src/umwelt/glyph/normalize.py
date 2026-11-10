"""Glyph normalization — grammar §5.

Canonical form: NFC; variation selectors U+FE0E / U+FE0F stripped (so a sun with and
without the emoji selector are one glyph); ZWJ sequences kept whole as ONE glyph;
skin-tone modifiers U+1F3FB..U+1F3FF stripped unless the scope declares them distinct.

``glyphs`` splits canonical text into grapheme-level glyph tokens with a small emoji-aware
splitter (no third-party dependency). Limits, honestly:

* It is not UAX #29. It handles: base + combining marks, base + skin tone (when kept),
  ZWJ chains (a ZWJ always joins the next code point to the cluster), regional-indicator
  PAIRS (flags), tag-character flag sequences, and keycap sequences (base + U+20E3).
* Hangul jamo, Indic conjuncts and other script-specific clustering are NOT modelled; a
  ZWJ in such scripts also joins, which is wrong for them. Glyphs are meant to be emoji
  and symbols.
* The splitter never reorders or drops code points; ``"".join(glyphs(t)) == canon(t)``.
"""
from __future__ import annotations

import unicodedata

VARIATION_SELECTORS = ("︎", "️")
ZWJ = "‍"
KEYCAP = "⃣"
_STRIP_VS = {ord(c): None for c in VARIATION_SELECTORS}
_STRIP_TONES = {cp: None for cp in range(0x1F3FB, 0x1F400)}


def _is_tone(ch: str) -> bool:
    return 0x1F3FB <= ord(ch) <= 0x1F3FF


def _is_regional(ch: str) -> bool:
    return 0x1F1E6 <= ord(ch) <= 0x1F1FF


def _is_tag(ch: str) -> bool:
    return 0xE0020 <= ord(ch) <= 0xE007F


def _is_extender(ch: str) -> bool:
    """Code points that attach to the cluster before them."""
    if _is_tone(ch) or _is_tag(ch) or ch == KEYCAP:
        return True
    return unicodedata.category(ch) in ("Mn", "Me", "Mc")


def canon(text: str, *, keep_skin_tones: bool = False) -> str:
    """Canonical form of ``text`` (grammar §5). Idempotent."""
    s = unicodedata.normalize("NFC", text).translate(_STRIP_VS)
    if not keep_skin_tones:
        s = s.translate(_STRIP_TONES)
    return unicodedata.normalize("NFC", s)


def glyphs(text: str, *, keep_skin_tones: bool = False) -> list[str]:
    """Split ``text`` (after ``canon``) into glyph tokens. Every character of the text,
    including ASCII punctuation and whitespace, becomes (part of) a token."""
    s = canon(text, keep_skin_tones=keep_skin_tones)
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        j = i + 1
        if _is_regional(s[i]) and j < n and _is_regional(s[j]):
            j += 1
        while j < n:
            ch = s[j]
            if ch == ZWJ:
                j += 1
                if j < n:
                    j += 1          # the joined code point
                continue
            if _is_extender(ch):
                j += 1
                continue
            break
        out.append(s[i:j])
        i = j
    return out
