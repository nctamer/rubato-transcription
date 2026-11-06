"""Python implementation of the unstable GNU libstdc++
``std::sort``; where equal keys occur its exact permutation decides ties (voice
hypotheses of equal cost, chord member order).  This introsort preserves compatible tie-breaking in the score reader."""


def cxx_sort(a, less):
    n = len(a)
    if n < 2:
        return a
    _introsort(a, 0, n, 2 * (n.bit_length() - 1), less)
    _final_insertion(a, 0, n, less)
    return a


def _introsort(a, first, last, depth, less):
    while last - first > 16:
        if depth == 0:
            _heapsort(a, first, last, less)
            return
        depth -= 1
        cut = _partition_pivot(a, first, last, less)
        _introsort(a, cut, last, depth, less)
        last = cut


def _partition_pivot(a, first, last, less):
    mid = first + (last - first) // 2
    _median_to_first(a, first, first + 1, mid, last - 1, less)
    return _unguarded_partition(a, first + 1, last, first, less)


def _median_to_first(a, result, x, y, z, less):
    if less(a[x], a[y]):
        if less(a[y], a[z]):
            a[result], a[y] = a[y], a[result]
        elif less(a[x], a[z]):
            a[result], a[z] = a[z], a[result]
        else:
            a[result], a[x] = a[x], a[result]
    elif less(a[x], a[z]):
        a[result], a[x] = a[x], a[result]
    elif less(a[y], a[z]):
        a[result], a[z] = a[z], a[result]
    else:
        a[result], a[y] = a[y], a[result]


def _unguarded_partition(a, first, last, pivot, less):
    while True:
        while less(a[first], a[pivot]):
            first += 1
        last -= 1
        while less(a[pivot], a[last]):
            last -= 1
        if not first < last:
            return first
        a[first], a[last] = a[last], a[first]
        first += 1


def _linear_insert(a, last, less):
    val = a[last]
    nxt = last - 1
    while less(val, a[nxt]):
        a[last] = a[nxt]
        last = nxt
        nxt -= 1
    a[last] = val


def _insertion(a, first, last, less):
    for i in range(first + 1, last):
        if less(a[i], a[first]):
            val = a[i]
            a[first + 1:i + 1] = a[first:i]
            a[first] = val
        else:
            _linear_insert(a, i, less)


def _final_insertion(a, first, last, less):
    if last - first > 16:
        _insertion(a, first, first + 16, less)
        for i in range(first + 16, last):
            _linear_insert(a, i, less)
    else:
        _insertion(a, first, last, less)


def _adjust_heap(a, base, hole, length, value, less):
    top = hole
    second = hole
    while second < (length - 1) // 2:
        second = 2 * (second + 1)
        if less(a[base + second], a[base + second - 1]):
            second -= 1
        a[base + hole] = a[base + second]
        hole = second
    if (length & 1) == 0 and second == (length - 2) // 2:
        second = 2 * (second + 1)
        a[base + hole] = a[base + second - 1]
        hole = second - 1
    parent = (hole - 1) // 2
    while hole > top and less(a[base + parent], value):
        a[base + hole] = a[base + parent]
        hole = parent
        parent = (hole - 1) // 2
    a[base + hole] = value


def _heapsort(a, first, last, less):
    length = last - first
    if length >= 2:
        parent = (length - 2) // 2
        while True:
            _adjust_heap(a, first, parent, length, a[first + parent], less)
            if parent == 0:
                break
            parent -= 1
    while last - first > 1:
        last -= 1
        value = a[last]
        a[last] = a[first]
        _adjust_heap(a, first, 0, last - first, value, less)


# The same sort with less(x, y) = x[0] < y[0], without a call per comparison.


def cxx_sort_k0(a):
    """cxx_sort with less(x, y) = x[0] < y[0] (same permutation, no call per compare)."""
    n = len(a)
    if n < 2:
        return a
    _introsort_k0(a, 0, n, 2 * (n.bit_length() - 1))
    _final_insertion_k0(a, 0, n)
    return a


def _introsort_k0(a, first, last, depth):
    while last - first > 16:
        if depth == 0:
            _heapsort_k0(a, first, last)
            return
        depth -= 1
        cut = _partition_pivot_k0(a, first, last)
        _introsort_k0(a, cut, last, depth)
        last = cut


def _partition_pivot_k0(a, first, last):
    mid = first + (last - first) // 2
    _median_to_first_k0(a, first, first + 1, mid, last - 1)
    return _unguarded_partition_k0(a, first + 1, last, first)


def _median_to_first_k0(a, result, x, y, z):
    if a[x][0] < a[y][0]:
        if a[y][0] < a[z][0]:
            a[result], a[y] = a[y], a[result]
        elif a[x][0] < a[z][0]:
            a[result], a[z] = a[z], a[result]
        else:
            a[result], a[x] = a[x], a[result]
    elif a[x][0] < a[z][0]:
        a[result], a[x] = a[x], a[result]
    elif a[y][0] < a[z][0]:
        a[result], a[z] = a[z], a[result]
    else:
        a[result], a[y] = a[y], a[result]


def _unguarded_partition_k0(a, first, last, pivot):
    while True:
        while a[first][0] < a[pivot][0]:
            first += 1
        last -= 1
        while a[pivot][0] < a[last][0]:
            last -= 1
        if not first < last:
            return first
        a[first], a[last] = a[last], a[first]
        first += 1


def _linear_insert_k0(a, last):
    val = a[last]
    nxt = last - 1
    while val[0] < a[nxt][0]:
        a[last] = a[nxt]
        last = nxt
        nxt -= 1
    a[last] = val


def _insertion_k0(a, first, last):
    for i in range(first + 1, last):
        if a[i][0] < a[first][0]:
            val = a[i]
            a[first + 1:i + 1] = a[first:i]
            a[first] = val
        else:
            _linear_insert_k0(a, i)


def _final_insertion_k0(a, first, last):
    if last - first > 16:
        _insertion_k0(a, first, first + 16)
        for i in range(first + 16, last):
            _linear_insert_k0(a, i)
    else:
        _insertion_k0(a, first, last)


def _adjust_heap_k0(a, base, hole, length, value):
    top = hole
    second = hole
    while second < (length - 1) // 2:
        second = 2 * (second + 1)
        if a[base + second][0] < a[base + second - 1][0]:
            second -= 1
        a[base + hole] = a[base + second]
        hole = second
    if (length & 1) == 0 and second == (length - 2) // 2:
        second = 2 * (second + 1)
        a[base + hole] = a[base + second - 1]
        hole = second - 1
    parent = (hole - 1) // 2
    while hole > top and a[base + parent][0] < value[0]:
        a[base + hole] = a[base + parent]
        hole = parent
        parent = (hole - 1) // 2
    a[base + hole] = value


def _heapsort_k0(a, first, last):
    length = last - first
    if length >= 2:
        parent = (length - 2) // 2
        while True:
            _adjust_heap_k0(a, first, parent, length, a[first + parent])
            if parent == 0:
                break
            parent -= 1
    while last - first > 1:
        last -= 1
        value = a[last]
        a[last] = a[first]
        _adjust_heap_k0(a, first, 0, last - first, value)
