"""Frozen one-sided memory rule; common raw pair generator is unchanged."""
from active_memory_support import prepare_memory
from withdrawal_guard_support import guard_scores


def prepare_guarded(local, centers, cache, root, radius, seen, joint, area_get, cap):
    record, arrays = prepare_memory(local, centers, cache, root, radius, seen, joint, area_get, cap)
    for kind in ('scalar', 'mscs'):
        record['memory_' + kind + '_scores'] = record[kind + '_scores']
        score, mask = guard_scores(root, record['candidates'], record['raw_' + kind + '_scores'],
                                   record['memory_' + kind + '_scores'])
        record[kind + '_scores'] = score.tolist()
    record['withdrawal'] = mask.tolist()
    return record, arrays
