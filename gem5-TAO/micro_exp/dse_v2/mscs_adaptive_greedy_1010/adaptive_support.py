"""Causal allocation between two unchanged finite-step ranking models."""
from pathlib import Path
import sys
import copy
import itertools
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASELINE = ROOT/'mscs_hybrid_greedy_1010'
PREVIOUS = ROOT/'mscs_jacobian_greedy_1010'
sys.path.insert(0, str(BASELINE))
from hybrid_support import (LocalModel as HybridLocalModel, PARAMS, STEPS, TOPK,
    DEFAULT_CKPT, NORM_CACHE, config_key, compute_w_min, legal, moves, Inputs,
    FeatureCache, load_model, exact, cpis, contributions, probabilities,
    choose as fixed_choose)

ARMS = ('adaptive',)
COMPARISON_ARMS = ('scalar_excess', 'mscs_logit', 'hybrid', 'adaptive')


class LocalModel(HybridLocalModel):
    def __init__(self, model, x, width, arm):
        assert arm == 'adaptive'
        super().__init__(model, x, width, 'hybrid')


def rank_feedback(root_cpi, scalar, mscs, actual):
    """Score both models on identical observed pairs, including the current root.

    Only verified CPI values enter actual. Area is deliberately absent: this is
    prediction-ranking feedback, not a learned feasibility predictor. Loss units
    are twice inversion loss: correct=0, predicted tie=1, reversed order=2.
    Ignore actual ties at the unchanged acceptance tolerance 1e-6 * root CPI.
    """
    arrays = [np.asarray(x, dtype=float) for x in (scalar, mscs, actual)]
    assert root_cpi > 0 and all(x.ndim == 1 for x in arrays)
    assert len({x.shape for x in arrays}) == 1 and all(np.isfinite(x).all() for x in arrays)
    scalar, mscs, actual = [np.r_[root_cpi, x] for x in arrays]
    tolerance = 1e-6*root_cpi
    def sign(a, b):
        return 1 if a-b > tolerance else -1 if a-b < -tolerance else 0
    errors = [0, 0]
    comparisons = []
    for i, j in itertools.combinations(range(len(actual)), 2):
        truth = sign(actual[i], actual[j])
        if truth == 0:
            continue
        predictions = [sign(x[i], x[j]) for x in (scalar, mscs)]
        loss = [0 if s == truth else 1 if s == 0 else 2 for s in predictions]
        errors = [a+b for a, b in zip(errors, loss)]
        comparisons.append(dict(pair=[i, j], actual_order=truth,
                                predicted_order=predictions, loss2=loss))
    return dict(loss2={'scalar': errors[0], 'mscs': errors[1]},
                comparisons=comparisons,
                discriminating_comparisons=sum(x['loss2'][0] != x['loss2'][1] for x in comparisons),
                tolerance=tolerance, pair_index_meaning='0=current root; 1,2=queried candidates in frozen order')


class RankAllocator:
    """Follow the cumulative ranking leader; tied leaders share the budget.

    State is private to one trajectory. No benchmark, parameter identity,
    counterfactual label, future round, area or discarded candidate truth enters
    allocation. There is no fitted threshold, forgetting rate or reward scaling.
    Sparse, selected-candidate feedback is NOT a statistical confidence bound.
    """
    def __init__(self):
        self.state = dict(loss2={'scalar': 0, 'mscs': 0}, completed_rounds=0,
                          comparisons=0, discriminating_comparisons=0)

    def snapshot(self):
        return copy.deepcopy(self.state)

    @property
    def mode(self):
        s, m = self.state['loss2']['scalar'], self.state['loss2']['mscs']
        return 'scalar_excess' if s < m else 'mscs_logit' if m < s else 'hybrid'

    def choose(self, scalar_scores, mscs_scores):
        return fixed_choose(self.mode, scalar_scores, mscs_scores)

    def observe(self, root_cpi, scalar, mscs, actual):
        result = rank_feedback(root_cpi, scalar, mscs, actual)
        for key in ('scalar', 'mscs'):
            self.state['loss2'][key] += result['loss2'][key]
        self.state['completed_rounds'] += 1
        self.state['comparisons'] += len(result['comparisons'])
        self.state['discriminating_comparisons'] += result['discriminating_comparisons']
        return result
