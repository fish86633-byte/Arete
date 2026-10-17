"""Exact canonical state SHA with bounded, invocation-local FIFO byte reuse.

Only immutable logical FIFO pieces are cached. Appending outside an old piece
cannot change its bytes; head consumption/replacement changes its key. RT is
always encoded afresh. Every call streams every canonical byte through SHA256.
Neither final hashes nor whole-state encodings are cached. Message encoding is
unmodified and does not use this cache.
"""
from __future__ import annotations

from collections import OrderedDict
from contextvars import ContextVar
import hashlib
import time
import weakref

from .fifo import CompactFIFO


MAX_ENCODED_BYTES = 64 * 1024 * 1024
MAX_ENTRIES = 32768
_current = ContextVar('pcqr_invocation_digest_cache', default=None)


class DigestCache:
    def __init__(self, max_bytes=MAX_ENCODED_BYTES, max_entries=MAX_ENTRIES):
        self.max_bytes, self.max_entries = int(max_bytes), int(max_entries)
        self.parts = OrderedDict()
        self.bytes = self.peak_bytes = self.peak_entries = 0
        self.hits = self.misses = self.evictions = self.hash_calls = 0
        self.streamed_bytes = 0
        self.closed = False

    def piece_bytes(self, piece, encode):
        # Keep the storage object itself in the key: object-id reuse cannot
        # accidentally hit stale bytes. The bounded cache owns these references
        # only until eviction or the enclosing Meter's charged finish.
        key = (piece.storage, piece.lo, piece.hi)
        found = self.parts.get(key)
        if found is not None:
            self.hits += 1
            self.parts.move_to_end(key)
            return found
        self.misses += 1
        encoded = encode(piece.view()).encode('utf-8')[1:-1]
        if len(encoded) <= self.max_bytes and self.max_entries > 0:
            while self.parts and (self.bytes+len(encoded) > self.max_bytes
                                  or len(self.parts) >= self.max_entries):
                _, evicted = self.parts.popitem(last=False)
                self.bytes -= len(evicted)
                self.evictions += 1
            self.parts[key] = encoded
            self.bytes += len(encoded)
            self.peak_bytes = max(self.peak_bytes, self.bytes)
            self.peak_entries = max(self.peak_entries, len(self.parts))
        return encoded

    def snapshot(self):
        return {'hash_calls': self.hash_calls, 'piece_hits': self.hits,
                'piece_misses': self.misses, 'piece_evictions': self.evictions,
                'streamed_canonical_bytes': self.streamed_bytes,
                'max_encoded_bytes': self.max_bytes, 'max_entries': self.max_entries,
                'peak_encoded_bytes': self.peak_bytes, 'peak_entries': self.peak_entries,
                'retained_encoded_bytes': self.bytes, 'retained_entries': len(self.parts),
                'closed': self.closed,
                'python_object_and_retained_storage_bytes': None,
                'scope': 'Cold for each Meter. Encoded-byte and entry limits are both enforced; Python/container/storage allocation is not an exact RSS bound. Every hash streams all canonical state bytes. Message dumps are unchanged.'}

    def close(self):
        wall, cpu = time.perf_counter(), time.process_time()
        self.parts.clear()
        self.bytes = 0
        self.closed = True
        return {'release_wall_s': time.perf_counter()-wall,
                'release_cpu_s': time.process_time()-cpu, **self.snapshot()}


def begin_cache(max_bytes=MAX_ENCODED_BYTES, max_entries=MAX_ENTRIES):
    cache = DigestCache(max_bytes, max_entries)
    # A lost/exception-abandoned Meter cannot make a context retain FIFO data.
    token = _current.set(weakref.ref(cache))
    return cache, token


def end_cache(cache, token):
    report = cache.close()
    _current.reset(token)
    return report


def active_cache():
    reference = _current.get()
    cache = reference() if reference is not None else None
    return cache if cache is not None and not cache.closed else None


def canonical_chunks(state, encode, cache=None):
    """Yield exactly support.dumps({'rt': ..., 'nrt_batches': ...}).encode()."""
    batches = state.nrt_batches
    if not isinstance(batches, CompactFIFO):
        # Legacy/list-like input domain is preserved, including its encoder
        # errors. Ordinary authoritative State always holds CompactFIFO.
        yield encode({'rt': state.rt, 'nrt_batches': batches}).encode('utf-8')
        return
    yield b'{"nrt_batches":['
    for cell_index, cell in enumerate(batches._cells):
        if cell_index:
            yield b','
        yield b'['
        first = True
        for piece in cell.pieces:
            if not len(piece):
                continue
            if not first:
                yield b','
            first = False
            yield (cache.piece_bytes(piece, encode) if cache is not None
                   else encode(piece.view()).encode('utf-8')[1:-1])
        yield b']'
    yield b'],"rt":'
    yield encode(state.rt).encode('utf-8')
    yield b'}'


def state_sha256(state, encode):
    cache = active_cache()
    if cache is not None:
        cache.hash_calls += 1
    digest = hashlib.sha256()
    size = 0
    for chunk in canonical_chunks(state, encode, cache):
        digest.update(chunk)
        size += len(chunk)
    if cache is not None:
        cache.streamed_bytes += size
    return digest.hexdigest()
