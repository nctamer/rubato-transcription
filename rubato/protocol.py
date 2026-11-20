"""Long-form chunking and stitching (pure Python / numpy).

A piece is cut into windows of `chunk_size` seconds every `hop_size` seconds. Each window is decoded on
its own; decoding of a window stops at the first timestamp token whose value exceeds the window's
timestamp limit (`max_generation_timestamp`; for the last window max(limit, its audio length + 0.01)).
The stitched transcript keeps, from every window but the last, the text up to the last timestamp
<= hop_size, shifts its timestamps by the window start and joins the pieces with spaces. The default
protocol (w40h20) uses 40 s windows every 20 s with a 20 s limit: each window keeps its first 20 s and
sees 20 s of audio after it.

These helpers define chunk boundaries, last-window handling, and timestamp stitching.
"""

import math
import re

import numpy as np

SR = 16000
LAST_AUDIO_TAIL = 1.0       # seconds of silence appended to the last window's audio
LAST_TAIL = 0.2             # cutoff slack for the last window's text
MIN_LAST_CHUNK_SEC = 10.0   # a last window shorter than this is dropped when the previous one covers the end

# The only windowing: 40 s windows every 20 s, each keeps its first 20 s (the 20 s timestamp ends the window), for
# every mode. Decoding is greedy.
W40H20 = dict(chunk_size=40.0, hop_size=20.0, max_generation_timestamp=20.0, merge_start_offset=0.0)

_TASK_TOKEN_RE = re.compile(
    r'<\|(?:no)?(?:pnc|itn|timestamp)\|>'
    r'|<\|(?:performance|score|beats|meter|piano_solo)\|>'
    r'|<\|continue\|>'
    r'|<\|(?:no)?diarize\|>'
    r'|<\|emo:[^|]+\|>'
)
_TIMESTAMP_RE = re.compile(r'<\|(\d+(?:\.\d+)?)\|>')


def strip_task_tokens(text):
    out = _TASK_TOKEN_RE.sub('', text)
    return re.sub(r'\s+', ' ', out).strip()


def shift_timestamps_by_offset(text, offset_seconds):
    def repl(m):
        val = float(m.group(1))
        new_val = val + offset_seconds
        return f'<|{new_val:.2f}|>' if '.' in m.group(1) else f'<|{int(round(new_val))}|>'
    return _TIMESTAMP_RE.sub(repl, text)


def extract_segment_up_to_time(transcript, max_time, *, exclusive_end=False):
    last_valid_end = 0
    for m in _TIMESTAMP_RE.finditer(transcript):
        t = float(m.group(1))
        if (t < max_time) if exclusive_end else (t <= max_time):
            last_valid_end = m.end()
        else:
            break
    if last_valid_end == 0 and not _TIMESTAMP_RE.search(transcript):
        return transcript.rstrip()
    return transcript[:last_valid_end].rstrip()


def extract_segment_with_timestamp_range(transcript, t_min, t_max):
    matches = [(m.start(), m.end(), float(m.group(1))) for m in _TIMESTAMP_RE.finditer(transcript)]
    in_range = [(i, s, e, t) for i, (s, e, t) in enumerate(matches) if t_min <= t <= t_max]
    if not in_range:
        return ''
    i0 = in_range[0][0]
    i1 = in_range[-1][0]
    seg_start = 0 if i0 == 0 else matches[i0 - 1][1]
    seg_end = matches[i1][1]
    segment = transcript[seg_start:seg_end].rstrip()
    lead = segment.lstrip()
    if lead.startswith('<|'):
        m_first = _TIMESTAMP_RE.match(lead)
        if m_first:
            segment = lead[m_first.end():].lstrip()
    return segment


def timestamp_values(vocab):
    """{token_id: seconds} for the numeric timestamp tokens."""
    out = {}
    for token_id, token in enumerate(vocab):
        m = re.match(r'^<\|(\d+(?:\.\d+)?)\|>$', token)
        if m:
            out[token_id] = float(m.group(1))
    return out


def n_chunks(n_samples, hop_size, sr=SR, min_last_chunk_seconds=None, chunk_size=None):
    if n_samples == 0:
        return 0
    hop_len = int(hop_size * sr)
    n = math.ceil(n_samples / hop_len)
    if min_last_chunk_seconds is not None and n > 1:
        last_chunk_start_samples = (n - 1) * hop_len
        last_chunk_duration_sec = (n_samples - last_chunk_start_samples) / sr
        if last_chunk_duration_sec < min_last_chunk_seconds:
            if chunk_size is not None:
                prev_chunk_end = (n - 2) * hop_size + chunk_size
                if prev_chunk_end >= n_samples / sr:
                    n = n - 1
            else:
                n = n - 1
    return n


def get_chunk(audio_np, chunk_idx, n_chunks_, frame_len, hop_len, last_chunk_tail=0):
    n_samples = len(audio_np)
    start = chunk_idx * hop_len
    end = start + frame_len
    is_last = (chunk_idx == n_chunks_ - 1)
    if end <= n_samples:
        return audio_np[start:end].copy()
    elif is_last:
        actual = n_samples - start
        length = min(actual + last_chunk_tail, frame_len)
        chunk = np.zeros(length, dtype=audio_np.dtype)
        if actual > 0:
            chunk[:actual] = audio_np[start:n_samples]
        return chunk
    else:
        chunk = np.zeros(frame_len, dtype=audio_np.dtype)
        actual = n_samples - start
        chunk[:actual] = audio_np[start:n_samples]
        return chunk


def plan_piece(n_samples, chunk_size, hop_size, merge_start_offset=0.0, sr=SR):
    """dict(n_chunks, starts, actual_durations, max_times, duration) of one piece."""
    duration = n_samples / sr
    n = n_chunks(n_samples, hop_size, sr, min_last_chunk_seconds=MIN_LAST_CHUNK_SEC, chunk_size=chunk_size)
    starts = [i * hop_size for i in range(n)]
    if n == 0:
        return dict(n_chunks=0, starts=[], actual_durations=[], max_times=[], duration=duration)
    actual = [min(chunk_size, duration - s) for s in starts]
    if merge_start_offset > 0 and n > 1:
        max_times = ([hop_size + merge_start_offset] + [merge_start_offset + hop_size] * (n - 2)
                     + [actual[-1] + LAST_TAIL])
    elif merge_start_offset > 0 and n == 1:
        max_times = [actual[-1] + LAST_TAIL]
    else:
        max_times = [hop_size] * (n - 1) + [actual[-1] + LAST_TAIL] if n > 1 else [actual[-1] + LAST_TAIL]
    return dict(n_chunks=n, starts=starts, actual_durations=actual, max_times=max_times, duration=duration)


def chunk_audio(audio_np, plan, chunk_size, hop_size, sr=SR):
    """[n_chunks, frame_len] float32 (each window zero-padded to frame_len) and true lengths [n_chunks]."""
    frame_len, hop_len = int(chunk_size * sr), int(hop_size * sr)
    n = plan['n_chunks']
    out = np.zeros((n, frame_len), dtype=np.float32)
    lens = np.zeros(n, dtype=np.int64)
    for c in range(n):
        ch = get_chunk(audio_np, c, n, frame_len, hop_len, last_chunk_tail=int(LAST_AUDIO_TAIL * sr))
        out[c, :len(ch)] = ch
        lens[c] = len(ch)
    return out, lens


def eos_thresholds(plan, max_generation_timestamp):
    """Per-window timestamp limit: a timestamp token above it ends the window's decoding."""
    base = float(max_generation_timestamp)
    n = plan['n_chunks']
    return [max(base, plan['actual_durations'][c] + 0.01) if c == n - 1 else base for c in range(n)]


def stitch_chunks(chunk_transcripts, start_seconds, chunk_durations, merge_start_offset=0.0):
    parts = []
    for i, (transcript, start) in enumerate(zip(chunk_transcripts, start_seconds)):
        max_time = chunk_durations[i]
        if merge_start_offset > 0 and i > 0:
            segment = extract_segment_with_timestamp_range(transcript, merge_start_offset, max_time)
        elif merge_start_offset > 0 and i == 0:
            segment = extract_segment_up_to_time(transcript, max_time, exclusive_end=True)
        else:
            segment = extract_segment_up_to_time(transcript, max_time)
        if segment.strip():
            parts.append(shift_timestamps_by_offset(segment, start))
    return ' '.join(p for p in parts if p.strip())


def merge_piece(chunk_texts_raw, plan, merge_start_offset=0.0):
    """Stitched transcript from per-window texts (task tokens are stripped from every window but the first)."""
    texts = [strip_task_tokens(t or '') if i > 0 else (t or '') for i, t in enumerate(chunk_texts_raw)]
    return stitch_chunks(texts, plan['starts'], plan['max_times'], merge_start_offset)


# Notes still sounding at a window cut lose their release with the discarded text. merge_piece_midi closes them
# with the release in the window's stopping stamp group, else the next window's leading release of the pitch,
# else at the cut.
_MIDI_TOK = re.compile(r'<\|CC64:(?:on|off)\|>|<\|vel:\d+\|>|<\|(\d+(?:\.\d+)?)\|>|([A-Ga-g]#?\d+)')


def _open_pitches(segment):
    """Pitch names (upper case) still sounding at the end of a midi-mode segment (release < onset at one stamp)."""
    events, pending = [], []
    for m in _MIDI_TOK.finditer(segment):
        if m.group(1):
            t = float(m.group(1))
            events += [(t, 0 if tok.islower() else 1, tok.upper()) for tok in pending]
            pending = []
        elif m.group(2):
            pending.append(m.group(2))
    events.sort(key=lambda e: (e[0], e[1]))
    active = set()
    for _, on, p in events:
        (active.add if on else active.discard)(p)
    return active


def _stop_group_releases(transcript, max_time):
    """(stamp, {PITCH}) of the releases in the first stamp group past max_time, or (None, set())."""
    last = 0
    for m in _TIMESTAMP_RE.finditer(transcript):
        if float(m.group(1)) > max_time:
            grp = transcript[last:m.start()]
            toks = [x.group(2) for x in _MIDI_TOK.finditer(grp) if x.group(2)]
            return float(m.group(1)), {t.upper() for t in toks if t[0].islower()}
        last = m.end()
    return None, set()


def _leading_releases(transcript):
    """{PITCH} released in a window before any onset of that pitch in it."""
    out, seen, pending = set(), set(), []
    for m in _MIDI_TOK.finditer(transcript):
        if m.group(1):
            for tok in pending:
                if tok[0].isupper():
                    seen.add(tok.upper())
                elif tok.upper() not in seen:
                    out.add(tok.upper())
            pending = []
        elif m.group(2):
            pending.append(m.group(2))
    return out


def merge_piece_midi(chunk_texts_raw, plan):
    """merge_piece for midi-mode text (merge_start_offset 0), closing the notes left open at every cut."""
    texts = [strip_task_tokens(t or '') if i > 0 else (t or '') for i, t in enumerate(chunk_texts_raw)]
    n = len(texts)
    parts = []
    for i, (tx, start) in enumerate(zip(texts, plan['starts'])):
        cut = plan['max_times'][i]
        seg = extract_segment_up_to_time(tx, cut)
        if i < n - 1:
            open_ = _open_pitches(seg)
            if open_:
                add = []
                t, rel = _stop_group_releases(tx, cut)
                own = sorted(open_ & rel)
                if own:
                    add.append(' '.join(p.lower() for p in own) + f'<|{t:.2f}|>')
                    open_ -= set(own)
                rest = sorted(open_ - _leading_releases(texts[i + 1]))
                if rest:
                    add.append(' '.join(p.lower() for p in rest) + f'<|{cut:.2f}|>')
                if add:
                    seg = (seg + ' ' + ' '.join(add)).strip()
        if seg.strip():
            parts.append(shift_timestamps_by_offset(seg, start))
    return ' '.join(parts)


# Score mode, per staff: a note open at a cut is closed by the next window's leading release of its pitch, else
# at the next window's first interval.
_SCORE_PITCH = re.compile(r'^([A-Ga-g])(#{1,2}|-{1,2})?(-?\d+)')
_SCORE_INTERVAL = re.compile(r'^\d+/\d+$')
_FENCE = re.compile(r'<\|[^|]*\|>')
_SEMI = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def _score_tokens(text):
    """[(kind, value, raw index)] with kind in staff / interval / on / off / other; stamps stripped."""
    out = []
    for k, raw in enumerate(text.split()):
        t = _FENCE.sub('', raw)
        if not t:
            continue
        if t.endswith(':') and len(t) > 1 and t[:-1].isalnum():
            out.append(('staff', t[:-1], k))
        elif _SCORE_INTERVAL.match(t):
            out.append(('interval', t, k))
        else:
            m = _SCORE_PITCH.match(t)
            if m and not t.startswith('|'):
                acc = m.group(2) or ''
                midi = (12 * (int(m.group(3)) + 1) + _SEMI[m.group(1).upper()]
                        + acc.count('#') - acc.count('-'))
                out.append(('on' if m.group(1).isupper() else 'off', (midi, t[:m.end()].lower()), k))
            else:
                out.append(('other', t, k))
    return out


def _score_open(segment, staff):
    """({(staff, midi): release spelling} open at the end, staff current at the end)."""
    open_ = {}
    for kind, v, _ in _score_tokens(segment):
        if kind == 'staff':
            staff = v
        elif kind == 'on':
            open_[(staff, v[0])] = v[1]
        elif kind == 'off':
            open_.pop((staff, v[0]), None)
    return open_, staff


def _score_leading_releases(segment, staff):
    out, seen = set(), set()
    for kind, v, _ in _score_tokens(segment):
        if kind == 'staff':
            staff = v
        elif kind == 'on':
            seen.add((staff, v[0]))
        elif kind == 'off' and (staff, v[0]) not in seen:
            out.add((staff, v[0]))
    return out


def merge_piece_score(chunk_texts_raw, plan):
    """merge_piece for score-mode text (merge_start_offset 0), closing the notes left open at every cut."""
    texts = [strip_task_tokens(t or '') if i > 0 else (t or '') for i, t in enumerate(chunk_texts_raw)]
    segs = [extract_segment_up_to_time(tx, plan['max_times'][i]) for i, tx in enumerate(texts)]
    staff = None
    open_ = {}
    for i in range(len(segs)):
        if i > 0 and open_:
            lead = _score_leading_releases(segs[i], staff)
            pending = {key: name for key, name in open_.items() if key not in lead}
            if pending:
                raw = segs[i].split()
                toks = _score_tokens(segs[i])
                first = next((k for kind, _, k in toks if kind == 'interval'), None)
                cur = staff
                for kind, v, k in toks:
                    if first is not None and k > first:
                        break
                    if kind == 'staff':
                        cur = v
                add = []
                for st in sorted({s for s, _ in pending}, key=str):
                    add.append(f'{st}:' if st else '')
                    add += sorted(name for (s, _), name in pending.items() if s == st)
                add = [a for a in add if a]
                nxt = raw[first + 1] if first is not None and first + 1 < len(raw) else ''
                last_written = next((a[:-1] for a in reversed(add) if a.endswith(':')), None)
                if cur is not None and cur != last_written and not (nxt.endswith(':') and nxt[:-1].isalnum()):
                    add.append(f'{cur}:')
                if first is None:
                    raw = raw + add
                else:
                    raw = raw[:first + 1] + add + raw[first + 1:]
                segs[i] = ' '.join(raw)
        open_, staff = _score_open(segs[i], staff) if i < len(segs) - 1 else ({}, staff)
    parts = [shift_timestamps_by_offset(s, st) for s, st in zip(segs, plan['starts']) if s.strip()]
    return ' '.join(parts)
