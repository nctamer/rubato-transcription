"""Voice (layer) estimation for one staff: a beam search over note groups.

Voice assignment strategy: notes sharing an onset form a group;
each hypothesis assigns every group member to a voice; costs reward melodic
continuity, chord fusion of same-ending notes and pitch-rank order (voice 0 on top)
and penalise opening voices. The validity rule is
that notes ending together share a voice; pruning retains the top eight candidates within a cost threshold of four.

Implementation notes:
* times are integer ticks (``unit`` per whole note), so ``float(Fraction(t, unit))``
  is ``t / unit``;
* a hypothesis keeps one immutable tuple per voice (``None`` = never opened), plus
  its open voices; voice states are built only for the candidates that survive;
* a group whose notes all end together (a single note or a chord: most groups) takes
  a fast path with the cost terms inlined; candidates whose cost is certain to
  exceed the best + PRUNE (by a lower bound of their cost) are counted but not
  built; when cost ties could make the unstable sort order matter, the
  group is redone in full and sorted with the port of that sort.
"""
from operator import itemgetter
from itertools import permutations
from math import gcd

from ._stdsort import cxx_sort, cxx_sort_k0

PROXIMITY = 0.01
CONTINUITY = -3.0
CONTINUITY_FALLOFF = 1.0
DURATION_SIMILARITY = -0.1
NEW_VOICE = 5.0
VOICE_PREFERENCE = 0.001
TEXTURE_CHANGE = 3.0
LATE_OPEN = 0.1
RANK = 3.0
COMMON_FATE = -1.0
SPLIT_FATE = 2.0
CONSONANCE = (-0.5, 0.5, 0.2, -0.2, -0.3, -0.1, 0.4, -0.4, -0.2, -0.3, 0.2, 0.4)
BEAM = 8
PRUNE = 4.0
MAX_VOICES = 8
BUDGET = 200000

_cost = itemgetter(0)
_EMPTY = (None,) * MAX_VOICES
# voice state: (last_off ticks, last_off_ev, psum, pcount, last_dur, last_chord); None = not opened
_OFF, _OFF_EV, _PSUM, _PCOUNT, _LDUR, _LCHORD = range(6)


def _term(L, ls, cnt, top, psum, dfirst, onset_t, onset_q, T):
    """Cost of putting a note group (cnt notes, best pitch rank top, pitch sum psum,
    duration dfirst of its first member) into voice L with state ls."""
    pref = L * VOICE_PREFERENCE
    rank_term = RANK * abs(float(L - top))
    if ls is None:
        return NEW_VOICE + pref + LATE_OPEN * onset_q + rank_term
    bavg = psum / cnt
    gap = (onset_t - ls[_OFF]) / T * 4.0
    if gap <= 0:
        cont = CONTINUITY
    elif gap < CONTINUITY_FALLOFF:
        cont = CONTINUITY * (1.0 - gap / CONTINUITY_FALLOFF)
    else:
        cont = 0.0
    pc = ls[_PCOUNT]
    prox = PROXIMITY * abs(bavg - (ls[_PSUM] / pc if pc else 60.0))
    dur = DURATION_SIMILARITY if (dfirst > 0 and abs(dfirst - ls[_LDUR]) < 1e-4) else 0.0
    tex = TEXTURE_CHANGE if (gap > 0 and (cnt >= 2) != ls[_LCHORD]) else 0.0
    return prox + cont + dur + pref + tex + rank_term


def _advance(states, run, lay, onset_q):
    """New voice-state tuple after assigning run (members laid out by lay)."""
    st = list(states)
    n = len(run)
    done = set()
    for i in range(n):
        L = lay[i]
        if L in done:
            continue
        done.add(L)
        cnt, psum = 0, 0.0
        boff = boff_ev = None
        for j in range(i, n):
            if lay[j] == L:
                cnt += 1
                psum += run[j][2]
                boff = run[j][1]
                boff_ev = run[j][4]
        old = st[L]
        if old is None:
            st[L] = (boff, boff_ev, 0.0 + psum, cnt, run[i][5] - onset_q, cnt >= 2)
        else:
            st[L] = (boff, boff_ev, old[_PSUM] + psum, old[_PCOUNT] + cnt, run[i][5] - onset_q, cnt >= 2)
    return tuple(st)


_PREF = tuple(L * VOICE_PREFERENCE for L in range(MAX_VOICES))
_RANK0 = tuple(RANK * abs(float(L - 0)) for L in range(MAX_VOICES))


# lower bound of the cost term of an active voice L: proximity >= 0, continuity >= -3,
# duration similarity >= -0.1, texture >= 0
_TERM_LB = tuple(CONTINUITY + DURATION_SIMILARITY + L * VOICE_PREFERENCE + RANK * float(L)
                 for L in range(MAX_VOICES))


def _extend1(beam, run, out, T, skip=False):
    """_extend for a run whose notes all end together (one fate class: a single note
    or a chord), with the cost terms inlined.  -> number of candidates, or None when
    some hypothesis needs the exhaustive search (left to _extend).

    skip: leave out candidates that are certain to be pruned (cost > best + PRUNE,
    best being at most the cheapest candidate so far) -- they are counted but not
    built.  The caller redoes the run without skipping when cost ties could make the
    unstable sort depend on them."""
    first = run[0]
    onset_t, onset_ev, onset_q = first[0], first[3], first[6]
    N = len(run)
    dq = first[5] - onset_q
    psum = 0.0
    for nt in run:
        psum += nt[2]
    bavg = psum / N
    chord = N >= 2
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            c = CONSONANCE[abs(run[i][2] - run[j][2]) % 12]
            c += COMMON_FATE
            pairs.append(c)
    # (a margin far above the rounding of these short float sums)
    big = max(abs(beam[0][0]), abs(beam[-1][0])) if beam else 0.0
    pair_lb = sum(pairs) - 1e-6 - 1e-9 * big
    late = LATE_OPEN * onset_q
    n = 0
    append = out.append
    PREF, RANK0, TLB = _PREF, _RANK0, _TERM_LB
    rb = float('inf') if skip else None    # cheapest candidate so far (+ PRUNE)
    for hidx, (hcost, hl, _h, act, ina) in enumerate(beam):
        nfree = 0
        mark = len(out)
        if skip:
            lim = rb - hcost - pair_lb
        for L, ls in act:
            if (ls[1] <= onset_ev) if ls[1] >= 0 else (ls[0] <= onset_t):
                nfree += 1
                if skip and TLB[L] > lim:
                    continue
                gap = (onset_t - ls[0]) / T * 4.0
                if gap <= 0:
                    cont = CONTINUITY
                elif gap < CONTINUITY_FALLOFF:
                    cont = CONTINUITY * (1.0 - gap / CONTINUITY_FALLOFF)
                else:
                    cont = 0.0
                pc = ls[3]
                prox = PROXIMITY * abs(bavg - (ls[2] / pc if pc else 60.0))
                dur = DURATION_SIMILARITY if (dq > 0 and abs(dq - ls[4]) < 1e-4) else 0.0
                tex = TEXTURE_CHANGE if (gap > 0 and chord != ls[5]) else 0.0
                local = 0.0
                local += prox + cont + dur + PREF[L] + tex + RANK0[L]
                for p in pairs:
                    local += p
                cost = hcost + local
                if skip:
                    if cost > rb:
                        continue
                    if cost + PRUNE < rb:
                        rb = cost + PRUNE
                        lim = rb - hcost - pair_lb
                append((cost, hidx, L))
        B = nfree
        if ina:
            for L in (ina[:1] if nfree else ina):
                B += 1
                local = 0.0
                local += NEW_VOICE + PREF[L] + late + RANK0[L]
                for p in pairs:
                    local += p
                cost = hcost + local
                if skip:
                    if cost > rb:
                        continue
                    if cost + PRUNE < rb:
                        rb = cost + PRUNE
                append((cost, hidx, L))
        elif not nfree:
            # Every voice is busy; fall back to voice 0
            local = 0.0
            local += _term(0, hl[0], N, 0, psum, dq, onset_t, onset_q, T)
            for p in pairs:
                local += p
            cost = hcost + local
            B = 1
            if not skip or cost <= rb:
                if skip and cost + PRUNE < rb:
                    rb = cost + PRUNE
                append((cost, hidx, 0))
        if N > 1 and float(B) ** N > 100000.0:
            del out[mark:]
            return None
        n += B
    return n


def _hyp(cost, st, hist):
    """Beam entry: (cost, voice states, history, active voices [(L, state)], first two
    voices never opened)."""
    act = []
    ina = []
    L = 0
    for ls in st:
        if ls is None:
            if len(ina) < 2:
                ina.append(L)
        else:
            act.append((L, ls))
        L += 1
    return (cost, st, hist, act, tuple(ina))


def _advance1h(cost, parent, run, L):
    """Beam entry after putting every note of run into voice L of the parent entry."""
    _c, states, hist, act, ina = parent
    st = list(states)
    old = st[L]
    psum = 0.0
    for nt in run:
        psum += nt[2]
    last = run[-1]
    first = run[0]
    n = len(run)
    if old is None:
        new = st[L] = (last[1], last[4], 0.0 + psum, n, first[5] - first[6], n >= 2)
        return _hyp(cost, tuple(st), (hist, [L] * n))
    new = st[L] = (last[1], last[4], old[2] + psum, old[3] + n, first[5] - first[6], n >= 2)
    return (cost, tuple(st), (hist, [L] * n), [(l, new if l == L else x) for l, x in act], ina)


def _extend(beam, run, out, emitted, T):
    """beam: list of (cost, states, history).  Appends (cost, beam index, lay) to out."""
    N = len(run)
    onset_t = run[0][0]
    onset_ev = run[0][3]
    onset_q = run[0][6]
    midi = [n[2] for n in run]
    off_ev = [n[4] for n in run]
    dur_q = [n[5] - onset_q for n in run]
    by_pitch = sorted(range(N), key=lambda a: (-midi[a], a))
    rank = [0] * N
    for r, a in enumerate(by_pitch):
        rank[a] = r
    distinct_ends = len(set(off_ev))

    # Fate classes group notes with the same offset ordinal
    cls = [-1] * N
    n_cls = 0
    for i in range(N - 1, -1, -1):
        for j in range(N - 1, i, -1):
            if cls[i] >= 0:
                break
            if off_ev[i] == off_ev[j]:
                cls[i] = cls[j]
        if cls[i] < 0:
            cls[i] = n_cls
            n_cls += 1
    # per class: cnt, top rank, pitch sum (in note order), first member's duration
    cinfo = []
    for c in range(n_cls):
        cnt, first, top, psum = 0, -1, 10 ** 9, 0.0
        for i in range(N):
            if cls[i] == c:
                if first < 0:
                    first = i
                cnt += 1
                psum += midi[i]
                top = min(top, rank[i])
        cinfo.append((cnt, top, psum, dur_q[first]))
    # pair terms of an assignment where every slot holds one fate class (the
    # other pairs add +0.0, which leaves a sum that is not -0.0 unchanged)
    inj_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            if off_ev[i] == off_ev[j]:
                chord = CONSONANCE[abs(midi[i] - midi[j]) % 12]
                chord += COMMON_FATE
                inj_pairs.append(chord)
    pchord = None

    tcache = {}            # (class, voice, id(voice state)) -> cost term
    for hidx, (hcost, hl, _hist, _act, _ina) in enumerate(beam):
        cand = []
        inactive = []
        L = 0
        for ls in hl:
            if ls is None:
                inactive.append(L)
            elif (ls[1] <= onset_ev) if ls[1] >= 0 else (ls[0] <= onset_t):
                cand.append(L)
            L += 1
        want = max(0, distinct_ends - len(cand)) + 1
        if inactive:
            cand.extend(inactive[:want])
        B = len(cand)
        if B == 0:
            cand.append(0)
            B = 1

        if float(B) ** N <= 100000.0:
            if n_cls > B:
                continue
            terms = []
            for c in range(n_cls):
                ci = cinfo[c]
                row = []
                for L in cand:
                    ls = hl[L]
                    k = (c, L, ls)
                    v = tcache.get(k)
                    if v is None:
                        v = tcache[k] = _term(L, ls, ci[0], ci[1], ci[2], ci[3], onset_t, onset_q, T)
                    row.append(v)
                terms.append(row)
            if n_cls == 1:
                t0 = terms[0]
                if inj_pairs:
                    for s in range(B):
                        local = 0.0
                        local += t0[s]
                        for p in inj_pairs:
                            local += p
                        out.append((hcost + local, hidx, cand[s]))
                else:
                    for s in range(B):
                        out.append((hcost + (0.0 + t0[s]), hidx, cand[s]))
                emitted[0] += B
                continue
            cnt = 0
            for perm in permutations(range(B), n_cls):
                local = 0.0
                for c in sorted(range(n_cls), key=perm.__getitem__):
                    local += terms[c][perm[c]]
                for p in inj_pairs:
                    local += p
                out.append((hcost + local, hidx, [cand[perm[cls[i]]] for i in range(N)]))
                cnt += 1
            emitted[0] += cnt
        else:
            if pchord is None:
                pchord, psplit = {}, {}
                for i in range(N):
                    for j in range(i + 1, N):
                        chord = CONSONANCE[abs(midi[i] - midi[j]) % 12]
                        split = 0.0
                        if off_ev[i] == off_ev[j]:
                            chord += COMMON_FATE
                            split += SPLIT_FATE
                        pchord[i, j] = chord
                        psplit[i, j] = split
                pair_list = list(pchord)

            def commit(sel):
                local = 0.0
                for s in range(B):
                    L = cand[s]
                    cnt, first, top, psum = 0, -1, 10 ** 9, 0.0
                    for i in range(N):
                        if sel[i] == s:
                            if first < 0:
                                first = i
                            cnt += 1
                            psum += midi[i]
                            top = min(top, rank[i])
                    if cnt == 0:
                        continue
                    local += _term(L, hl[L], cnt, top, psum, dur_q[first], onset_t, onset_q, T)
                for ij in pair_list:
                    local += pchord[ij] if sel[ij[0]] == sel[ij[1]] else psplit[ij]
                out.append((hcost + local, hidx, [cand[sel[i]] for i in range(N)]))
                emitted[0] += 1

            sel = [0] * N
            slot_used = [False] * B
            slot_off = [None] * B
            overflow = [False]

            def rec(i):
                if overflow[0]:
                    return
                if i == N:
                    commit(sel)
                    if emitted[0] > BUDGET:
                        overflow[0] = True
                    return
                fate_s = -1
                for s in range(B):
                    if slot_used[s] and slot_off[s] == off_ev[i]:
                        fate_s = s
                        break

                def try_s(s):
                    wu, pv = slot_used[s], slot_off[s]
                    slot_used[s], slot_off[s], sel[i] = True, off_ev[i], s
                    rec(i + 1)
                    slot_used[s], slot_off[s] = wu, pv

                if fate_s >= 0:
                    try_s(fate_s)
                else:
                    for s in range(B):
                        if slot_used[s] and slot_off[s] != off_ev[i]:
                            continue
                        try_s(s)
                        if overflow[0]:
                            return
            rec(0)
            if overflow[0]:
                return False
    return True


def _greedy(notes):
    n = len(notes)
    order = sorted(range(n), key=lambda a: (notes[a][0], notes[a][1]))
    layer = [0] * n
    ends = []
    k = 0
    while k < n:
        e = k
        while e < n and notes[order[e]][0] == notes[order[k]][0] and notes[order[e]][1] == notes[order[k]][1]:
            e += 1
        li = next((c for c, end in enumerate(ends) if end <= notes[order[k]][0]), -1)
        if li < 0:
            li = len(ends)
            ends.append(0)
        for x in range(k, e):
            layer[order[x]] = li
        ends[li] = notes[order[k]][1]
        k = e
    return layer


def _best(nxt):
    """Candidate selection uses an unstable introsort and keeps the
    first BEAM.  When the BEAM smallest costs are distinct and the next one differs,
    every sort puts the same candidates, in the same order, in front."""
    s = sorted(nxt, key=_cost)
    if len(s) <= 16:
        return s[:BEAM]          # the introsort is a (stable) insertion sort here
    top = s[:BEAM + 1]
    for i in range(len(top) - 1):
        if top[i][0] == top[i + 1][0]:
            cxx_sort_k0(nxt)
            return nxt[:BEAM]
    return s[:BEAM]


def assign_voices(notes, unit=None):
    """notes: list of (onset, offset, midi, onset_event, offset_event) -> voice per note.

    onset / offset: Fractions (whole notes), or integer ticks of 1/unit whole note
    when unit is given; onset_event / offset_event are the ordinals of the clock
    positions in the piece (every interval token advances them), which decide
    simultaneity."""
    n = len(notes)
    if n == 0:
        return []
    key = [(nt[3], nt[4], nt[2]) for nt in notes]
    if len(set(key)) == n:
        order = sorted(range(n), key=key.__getitem__)
    else:
        order = cxx_sort(list(range(n)), lambda a, b: key[a] < key[b])
    runs = []
    i = 0
    while i < n:
        j = i
        ev0 = notes[order[i]][3]
        while j < n and notes[order[j]][3] == ev0:
            j += 1
        runs.append((i, j))
        i = j
    # single-stream bypass: every group is one chord and starts after all before it ended
    single, prev_off_ev, has = True, -1, False
    for rb, re_ in runs:
        first = notes[order[rb]]
        if notes[order[re_ - 1]][4] != first[4]:
            single = False
            break
        if has and prev_off_ev >= 0 and first[3] < prev_off_ev:
            single = False
            break
        prev_off_ev = max(prev_off_ev, first[4])
        has = True
    if single:
        return [0] * n
    # integer ticks: T = LCM of the denominators; t / T == float(the Fraction)
    if unit is None:
        T = 1
        for nt in notes:
            for x in (nt[0], nt[1]):
                d = x.denominator
                if T % d:
                    T = T * d // gcd(T, d)
    else:
        T = unit
    ticks = []
    for on, off, m, on_ev, off_ev in notes:
        if unit is None:
            on_t = on.numerator * (T // on.denominator)
            off_t = off.numerator * (T // off.denominator)
        else:
            on_t, off_t = on, off
        # (onset, offset, midi, onset_ev, offset_ev, offset_q, onset_q)
        ticks.append((on_t, off_t, m, on_ev, off_ev, off_t / T * 4.0, on_t / T * 4.0))
    beam = [_hyp(0.0, _EMPTY, None)]
    emitted = [0]
    for rb, re_ in runs:
        nxt = []
        run = [ticks[order[k]] for k in range(rb, re_)]
        if re_ - rb == 1 or run[0][4] == run[-1][4] and all(nt[4] == run[0][4] for nt in run):
            c = _extend1(beam, run, nxt, T, True)
            if c is not None:
                emitted[0] += c
                if not nxt:
                    return _greedy(notes)
                s = sorted(nxt, key=_cost)
                tie = False
                prev = None
                for x in s[:BEAM + 1]:
                    if x[0] == prev:
                        tie = True
                        break
                    prev = x[0]
                if tie:
                    # cost ties: rebuild the full candidate list, use the full introsort
                    nxt = []
                    _extend1(beam, run, nxt, T)
                    s = _best(nxt)
                nxt = s[:BEAM]
                best = nxt[0][0]
                while len(nxt) > 1 and nxt[-1][0] > best + PRUNE:
                    nxt.pop()
                beam = [_advance1h(cost, beam[hidx], run, L) for cost, hidx, L in nxt]
                continue
            nxt = []
        if not _extend(beam, run, nxt, emitted, T) or not nxt:
            return _greedy(notes)
        nxt = _best(nxt)
        best = nxt[0][0]
        while len(nxt) > 1 and nxt[-1][0] > best + PRUNE:
            nxt.pop()
        onset_q = run[0][6]
        nb = []
        for cost, hidx, lay in nxt:
            if lay.__class__ is int:
                lay = [lay] * len(run)
            nb.append(_hyp(cost, _advance(beam[hidx][1], run, lay, onset_q), (beam[hidx][2], lay)))
        beam = nb
    best = beam[0]
    for c in beam:
        if c[0] < best[0]:
            best = c
    seq = []
    h = best[2]
    while h is not None:
        h, lay = h
        seq.append(lay)
    seq.reverse()
    flat = [x for lay in seq for x in lay]
    out = [0] * n
    for k in range(n):
        out[order[k]] = flat[k]
    return out
