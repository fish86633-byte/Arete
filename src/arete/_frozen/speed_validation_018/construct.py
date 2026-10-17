"""Private constructor adapter; frozen 010/017 constructor bodies are intact."""
from __future__ import annotations

from arete._frozen.speed_validation_017 import construct as original_private
from arete._frozen.speed_validation_018.scorer import BatchedUpperScorer


def construct_proposal(Q0, A, channel, tpg_plan, meter, *, variant,
                       abs_start=0, oracle_callback=None, implementation='UB'):
    """Reuse U's constructor with only its private scorer class substituted.

Used in independent serial worker processes.  The class alias is restored even
on a budget exception or correctness error.  No default/core source is changed,
and no cache is transferred between scorer instances or planning attempts.
    """
    if implementation != 'UB':
        raise ValueError('only the preregistered UB implementation is supported')
    previous = original_private.UpperScorer
    try:
        original_private.UpperScorer = BatchedUpperScorer
        proposal, record = original_private.construct_proposal(
            Q0, A, channel, tpg_plan, meter, variant=variant, abs_start=abs_start,
            implementation='U', oracle_callback=oracle_callback)
    except Exception as error:
        failed_record = getattr(error, 'construction_record', None)
        if isinstance(failed_record, dict):
            failed_record['private_implementation'] = 'UB'
            failed_record['private_rt_age_batch_cache'] = True
        raise
    finally:
        original_private.UpperScorer = previous
    record['private_implementation'] = 'UB'
    record['private_rt_age_batch_cache'] = True
    return proposal, record
