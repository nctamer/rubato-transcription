"""ids -> text for the model's BPE vocabulary, without the sentencepiece library.

Reproduces SentencePiece's decoding for this model (identity normaliser, add_dummy_prefix, escaped
whitespace, byte fallback): pieces are concatenated with U+2581 replaced by a space; the leading U+2581 of the
first (non-control) piece is dropped; runs of byte pieces <0xNN> are decoded as UTF-8 (invalid bytes ->
U+FFFD); the unknown piece decodes to " ⁇ ".
"""

SPACE = "▁"
NORMAL, UNKNOWN, CONTROL, USER_DEFINED, UNUSED, BYTE = 1, 2, 3, 4, 5, 6


def _utf8_decode_spm(b):
    """SentencePiece's byte-run decoding: valid UTF-8 characters as is, every byte of an invalid sequence as
    U+FFFD."""
    out = []
    i, n = 0, len(b)
    while i < n:
        c = b[i]
        if c < 0x80:
            ln = 1
        elif 0xC2 <= c <= 0xDF:
            ln = 2
        elif 0xE0 <= c <= 0xEF:
            ln = 3
        elif 0xF0 <= c <= 0xF4:
            ln = 4
        else:
            ln = 0
        if ln and i + ln <= n:
            try:
                out.append(bytes(b[i:i + ln]).decode("utf-8"))
                i += ln
                continue
            except UnicodeDecodeError:
                pass
        out.append("�")
        i += 1
    return "".join(out)


class Tokenizer:
    def __init__(self, pieces, types, unk_surface=" ⁇ ", add_dummy_prefix=True, remove_extra_whitespaces=False):
        self.pieces = list(pieces)
        self.types = list(types)
        self.unk_surface = unk_surface
        self.strip_bos_ws = bool(add_dummy_prefix or remove_extra_whitespaces)
        self.byte_val = {i: int(p[3:5], 16) for i, (p, t) in enumerate(zip(self.pieces, self.types)) if t == BYTE}
        self.piece_to_id = {p: i for i, p in enumerate(self.pieces)}

    @property
    def vocab(self):
        return self.pieces

    def __len__(self):
        return len(self.pieces)

    def ids_to_text(self, ids):
        text = []
        is_bos_ws = True
        buf = bytearray()
        for i in ids:
            i = int(i)
            if i in self.byte_val:
                buf.append(self.byte_val[i])
                continue
            if buf:
                text.append(_utf8_decode_spm(buf))
                buf = bytearray()
                is_bos_ws = False
            t = self.types[i]
            p = self.pieces[i]
            if t == CONTROL:
                s = ""
            elif t == UNKNOWN:
                s = self.unk_surface
            else:
                if is_bos_ws and self.strip_bos_ws and p.startswith(SPACE):
                    p = p[len(SPACE):]
                s = p.replace(SPACE, " ")
            text.append(s)
            if t != CONTROL:
                is_bos_ws = False
        if buf:
            text.append(_utf8_decode_spm(buf))
        return "".join(text)
