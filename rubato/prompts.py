"""Fixed decoder prompts of the three modes (the model's own task tokens, in the order the model was trained
with). The weights file stores the corresponding ids; PROMPT_IDS are the ids for the released vocabulary,
checked against the training pipeline's prompt builder."""

_HEAD = ["▁", "<|startofcontext|>", "<|startoftranscript|>", "<|emo:undefined|>"]
_TAIL = ["<|timestamp|>", "<|nodiarize|>"]

PROMPT_TOKENS = {
    # printed score (notation) with timestamps
    "score": _HEAD + ["<|score|>", "<|piano_solo|>", "<|pnc|>", "<|noitn|>"] + _TAIL,
    # performance notes with velocities (MIDI-like) with timestamps
    "midi": _HEAD + ["<|performance|>", "<|piano_solo|>", "<|nopnc|>", "<|itn|>"] + _TAIL,
    # beats and downbeats with timestamps
    "beats": _HEAD + ["<|performance|>", "<|beats|>", "<|nopnc|>", "<|noitn|>"] + _TAIL,
}

PROMPT_IDS = {
    "score": [4429, 4, 3, 13, 19, 23, 5, 8, 9, 12],
    "midi": [4429, 4, 3, 13, 18, 23, 6, 7, 9, 12],
    "beats": [4429, 4, 3, 13, 18, 20, 6, 8, 9, 12],
}

MODES = tuple(PROMPT_TOKENS)
