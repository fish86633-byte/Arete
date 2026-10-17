"""Compact persistent NRT FIFO storage, with the original JSON representation.

Each cell consists of immutable logical views of contiguous birth/count arrays.
Array rows visible to any view are never overwritten. A private append-only
tail may grow outside every older view; diverging branches copy at most one
128-row tail. Assignments copy only their affected piece and prefix deletion
only advances views. Copying a complete queue copies M references, not N batches.

No batches are coalesced, including adjacent batches with the same birthday.
The constructor retains the old validation/filtering rules. Python integers
outside fixed-width ranges retain an object-array fallback instead of narrowing
the old accepted input domain. That fallback's array byte count covers pointer
payload only, not the Python integer allocations.
"""
from __future__ import annotations

from bisect import bisect_right
from contextlib import nullcontext
from dataclasses import dataclass
import operator

import numpy as np


CHUNK_ROWS = 128


def _part(telemetry, name):
    return telemetry.part(name) if telemetry is not None else nullcontext()


def _copied(telemetry, arrays, label):
    if telemetry is None:
        return
    numeric = sum(a.nbytes for a in arrays if a is not None and a.dtype.kind != 'O')
    objects = sum(a.nbytes for a in arrays if a is not None and a.dtype.kind == 'O')
    if numeric:
        telemetry.copied(int(numeric), label)
    if objects:
        telemetry.copied(int(objects), label+'_object_pointer_payload')


def _old_normalize(row):
    """Exact former State constructor semantics for non-integer-array inputs."""
    batches = []
    previous = None
    for birth, count in row:
        if int(birth) != birth or int(count) != count or count < 0:
            raise ValueError('FIFO birth/count must be integers with nonnegative count')
        if previous is not None and birth < previous:
            raise ValueError('NRT FIFO batches must be in arrival order')
        if count:
            batches.append((int(birth), int(count)))
        previous = birth
    return batches


def _integer_dtype(pairs):
    limit = np.iinfo(np.int64)
    return (np.int64 if all(limit.min <= value <= limit.max for pair in pairs for value in pair)
            else object)


def _validated_arrays(rows, telemetry=None):
    """Convert each decoded cell once on the ordinary JSON-integer path.

    Timing contexts cover whole passes, not individual batches/cells. Unusual
    accepted input types use the former scalar validation before conversion.
    Validation of arrival order includes zero-count entries before filtering.
    """
    arrays = []
    with _part(telemetry, 'array_conversion'):
        for row in rows:
            source = list(row) if isinstance(row, CellFIFO) else row
            try:
                arrays.append(np.array(source, copy=True))
            except (TypeError, ValueError, OverflowError):
                arrays.append(None)
    _copied(telemetry, arrays, 'decoded_nrt_cell_arrays')
    edits = []
    with _part(telemetry, 'item_validation'):
        for m, (row, array) in enumerate(zip(rows, arrays)):
            if array is not None and array.ndim == 1 and array.size == 0:
                edits.append((m, 'empty', None))
                continue
            if (array is not None and array.ndim == 2 and array.shape[1] == 2
                    and array.dtype.kind in 'iub'):
                if np.any(array[:, 1] < 0):
                    raise ValueError('FIFO birth/count must be integers with nonnegative count')
                if np.any(array[1:, 0] < array[:-1, 0]):
                    raise ValueError('NRT FIFO batches must be in arrival order')
                keep = array[:, 1] != 0
                if not np.all(keep):
                    edits.append((m, 'filter_bool' if array.dtype.kind == 'b' else 'filter', keep))
                elif array.dtype.kind == 'b':
                    edits.append((m, 'cast_bool', None))
            else:
                edits.append((m, 'scalar', _old_normalize(row)))
    converted = []
    with _part(telemetry, 'array_conversion'):
        for m, action, value in edits:
            if action == 'empty':
                replacement = np.empty((0, 2), dtype=np.int64)
            elif action == 'filter':
                replacement = arrays[m][value]
            elif action == 'filter_bool':
                # Boolean fields were cast to int by the former constructor.
                replacement = np.asarray(arrays[m][value], dtype=np.int64)
            elif action == 'cast_bool':
                replacement = np.asarray(arrays[m], dtype=np.int64)
            else:
                replacement = np.array(value, dtype=_integer_dtype(value)).reshape(-1, 2)
            arrays[m] = replacement
            converted.append(replacement)
    _copied(telemetry, converted, 'nrt_normalization_arrays')
    return arrays


class _Storage:
    __slots__ = ('data', 'used', 'appendable')

    def __init__(self, data, used, appendable=False):
        self.data = data
        self.used = int(used)
        self.appendable = bool(appendable)


@dataclass(frozen=True, slots=True)
class _Piece:
    storage: _Storage
    lo: int
    hi: int

    def __len__(self):
        return self.hi-self.lo

    def view(self):
        return self.storage.data[self.lo:self.hi]


@dataclass(frozen=True, slots=True)
class _Cell:
    pieces: tuple
    ends: tuple
    length: int
    total: int


_EMPTY = _Cell((), (), 0, 0)


def _cell(pieces, total):
    ends = []
    length = 0
    for piece in pieces:
        length += len(piece)
        ends.append(length)
    return _Cell(tuple(pieces), tuple(ends), length, int(total))


def _from_array(array):
    array.flags.writeable = False
    if not len(array):
        return _EMPTY
    storage = _Storage(array, len(array))
    pieces = tuple(_Piece(storage, lo, min(lo+CHUNK_ROWS, len(array)))
                   for lo in range(0, len(array), CHUNK_ROWS))
    # Python integer totals preserve the former sum() overflow behavior. The
    # normal nonnegative int64 path can sum vectorially after a safe bound.
    if (array.dtype.kind in 'iu' and int(array[:, 1].max(initial=0))*len(array)
            <= np.iinfo(np.int64).max):
        total = int(np.sum(array[:, 1], dtype=np.int64))
    else:
        total = sum(int(count) for count in array[:, 1])
    return _cell(pieces, total)


def _pair(value):
    birth, count = value
    if int(birth) != birth or int(count) != count or count < 0:
        raise ValueError('FIFO birth/count must be integers with nonnegative count')
    return int(birth), int(count)


def _fits(dtype, pair):
    if dtype.kind == 'O':
        return True
    limits = np.iinfo(dtype)
    return all(limits.min <= value <= limits.max for value in pair)


def _new_tail(pair, previous=None):
    dtype = np.dtype(np.int64) if previous is None else previous.storage.data.dtype
    if not _fits(dtype, pair):
        dtype = np.dtype(object)
    data = np.empty((CHUNK_ROWS, 2), dtype=dtype)
    used = len(previous) if previous is not None else 0
    if previous is not None:
        data[:used] = previous.view()
    data[used] = pair
    return _Piece(_Storage(data, used+1, appendable=True), 0, used+1)


class CellFIFO:
    """List-like cell handle; mutations replace only this owner's cell value."""
    __slots__ = ('_owner', '_index')

    def __init__(self, owner, index):
        self._owner, self._index = owner, index

    @property
    def _cell(self):
        return self._owner._cells[self._index]

    def __len__(self):
        return self._cell.length

    def __iter__(self):
        for piece in self._cell.pieces:
            for birth, count in piece.view():
                yield int(birth), int(count)

    def _locate(self, index):
        cell = self._cell
        index = operator.index(index)
        if index < 0:
            index += cell.length
        if index < 0 or index >= cell.length:
            raise IndexError('FIFO index out of range')
        piece_index = bisect_right(cell.ends, index)
        previous = cell.ends[piece_index-1] if piece_index else 0
        return piece_index, index-previous

    def __getitem__(self, index):
        if isinstance(index, slice):
            return list(self)[index]
        which, offset = self._locate(index)
        piece = self._cell.pieces[which]
        birth, count = piece.storage.data[piece.lo+offset]
        return int(birth), int(count)

    def append(self, value, *, exact_metadata=False):
        pair = _pair(value)
        cell = self._cell
        pieces = cell.pieces
        if pieces and len(pieces[-1]) < CHUNK_ROWS:
            tail = pieces[-1]
            storage = tail.storage
            if (storage.appendable and tail.hi == storage.used
                    and storage.used < len(storage.data) and _fits(storage.data.dtype, pair)):
                # Every existing immutable view ends at or before `used`.
                # The new row was inaccessible to all earlier snapshots.
                storage.data[storage.used] = pair
                storage.used += 1
                new_tail = _Piece(storage, tail.lo, tail.hi+1)
            else:
                new_tail = _new_tail(pair, tail)
            pieces = pieces[:-1]+(new_tail,)
            if exact_metadata:
                ends = cell.ends[:-1]+(cell.length+1,)
        else:
            pieces = pieces+(_new_tail(pair),)
            if exact_metadata:
                ends = cell.ends+(cell.length+1,)
        if exact_metadata:
            # Appending exactly one batch changes only the final cumulative
            # endpoint. Keep the same storage/tail COW path and Python-int
            # totals without rescanning every unchanged 128-row piece.
            value = _Cell(pieces, ends, cell.length+1, cell.total+pair[1])
        else:
            value = _cell(pieces, cell.total+pair[1])
        self._owner._cells[self._index] = value

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            raise TypeError('FIFO slice assignment is not used by the queue contract')
        pair = _pair(value)
        which, offset = self._locate(index)
        cell = self._cell
        piece = cell.pieces[which]
        old_count = int(piece.storage.data[piece.lo+offset, 1])
        dtype = piece.storage.data.dtype
        if not _fits(dtype, pair):
            dtype = np.dtype(object)
        data = np.array(piece.view(), dtype=dtype, copy=True)
        data[offset] = pair
        data.flags.writeable = False
        replacement = _Piece(_Storage(data, len(data)), 0, len(data))
        pieces = cell.pieces[:which]+(replacement,)+cell.pieces[which+1:]
        self._owner._cells[self._index] = _Cell(pieces, cell.ends, cell.length,
                                               cell.total-old_count+pair[1])

    def __delitem__(self, index):
        if not isinstance(index, slice):
            raise TypeError('FIFO deletion requires a prefix slice')
        cell = self._cell
        start, stop, step = index.indices(cell.length)
        if start != 0 or step != 1:
            raise ValueError('FIFO deletion requires a prefix slice')
        if stop == 0:
            return
        if stop == cell.length:
            self._owner._cells[self._index] = _EMPTY
            return
        removed = 0
        pieces = []
        left = stop
        for piece in cell.pieces:
            take = min(left, len(piece))
            if take:
                removed += sum(int(count) for count in piece.storage.data[piece.lo:piece.lo+take, 1])
            left -= take
            if take < len(piece):
                pieces.append(_Piece(piece.storage, piece.lo+take, piece.hi))
        self._owner._cells[self._index] = _cell(pieces, cell.total-removed)

    def to_json(self):
        rows = []
        for piece in self._cell.pieces:
            rows.extend(piece.view().tolist())
        return rows

    def __eq__(self, other):
        if isinstance(other, CellFIFO):
            if self._cell is other._cell:
                return True
            if self._cell.length != other._cell.length or self._cell.total != other._cell.total:
                return False
            return all(a == b for a, b in zip(self, other))
        if not isinstance(other, list) or len(self) != len(other):
            return False
        return all(a == b for a, b in zip(self, other))


class CompactFIFO:
    """Outer cell collection, persistent copying and old-format JSON adapter."""
    __slots__ = ('_cells',)

    def __init__(self, rows, telemetry=None):
        # The outer list contains only M references. It prevents a generator or
        # caller outer-list mutation from changing inputs between passes.
        with _part(telemetry, 'nrt_fifo_construction'):
            rows = list(rows)
        arrays = _validated_arrays(rows, telemetry)
        with _part(telemetry, 'nrt_fifo_construction'):
            self._cells = [_from_array(array) for array in arrays]
        if telemetry is not None:
            telemetry.set_fifo_batches(sum(cell.length for cell in self._cells))

    def copy(self):
        result = object.__new__(CompactFIFO)
        result._cells = self._cells.copy()
        return result

    def _append_verified_arrivals(self, counts, absolute_slot):
        """Append one batch per nonzero, already-validated int64 arrival.

        State.add owns public validation. The common path avoids constructing
        a CellFIFO handle and rechecking each trusted count against int64.
        Narrow/object storage and unusual birth values retain CellFIFO.append.
        Existing visible rows are never modified, including shared snapshots.
        """
        indices = np.flatnonzero(counts)
        if not len(indices):
            return
        # Nonstandard conversion objects retain the former per-cell int()
        # calls, errors and side effects. Ordinary birth integers convert once.
        if type(absolute_slot) is not int:
            for m in indices:
                self[int(m)].append((int(absolute_slot), int(counts[m])), exact_metadata=True)
            return
        birth = int(absolute_slot)
        if not -(1 << 63) <= birth < (1 << 63):
            for m in indices:
                self[int(m)].append((birth, int(counts[m])), exact_metadata=True)
            return
        cells = self._cells
        int64_dtype = np.dtype(np.int64)
        for m in indices:
            count = int(counts[m])
            cell = cells[m]
            pieces = cell.pieces
            if pieces and pieces[-1].storage.data.dtype != int64_dtype:
                self[int(m)].append((birth, count), exact_metadata=True)
                continue
            if pieces and len(pieces[-1]) < CHUNK_ROWS:
                tail = pieces[-1]
                storage = tail.storage
                if (storage.appendable and tail.hi == storage.used
                        and storage.used < len(storage.data)):
                    # birth and count are both certified int64; the appended
                    # row lies outside every existing immutable piece view.
                    storage.data[storage.used] = (birth, count)
                    storage.used += 1
                    new_tail = _Piece(storage, tail.lo, tail.hi + 1)
                else:
                    new_tail = _new_tail((birth, count), tail)
                pieces = pieces[:-1] + (new_tail,)
                ends = cell.ends[:-1] + (cell.length + 1,)
            else:
                pieces = pieces + (_new_tail((birth, count)),)
                ends = cell.ends + (cell.length + 1,)
            cells[m] = _Cell(pieces, ends, cell.length + 1, cell.total + count)

    def totals(self):
        # dtype conversion raises on out-of-range Python integers as the former
        # np.asarray([sum(...) ...], dtype=int64) property did.
        return np.fromiter((cell.total for cell in self._cells), dtype=np.int64,
                           count=len(self._cells))

    @property
    def batch_count(self):
        return sum(cell.length for cell in self._cells)

    def __len__(self):
        return len(self._cells)

    def __iter__(self):
        return (CellFIFO(self, m) for m in range(len(self._cells)))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [CellFIFO(self, m) for m in range(*index.indices(len(self)))]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError('FIFO cell index out of range')
        return CellFIFO(self, index)

    def to_json(self):
        return [row.to_json() for row in self]

    def __eq__(self, other):
        if not isinstance(other, (CompactFIFO, list)) or len(self) != len(other):
            return False
        return all(a == b for a, b in zip(self, other))
