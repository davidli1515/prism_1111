"""Memoize identical distribution inputs; keep every feature computation exact.

No architecture-slot shortcut and no hand-written dependency approximation.
The original encoder executes all of its formulas. Only repeated calls to its
pure encode_distribution function reuse an array for identical dtype, shape,
and input bytes. A private function globals dict avoids monkey-patching the
original module or affecting other experiments in this process.
"""
from collections import OrderedDict
from types import FunctionType
import time
import numpy as np
import torch
import secant_support  # Establish original module paths before importing core.
from core import Inputs
from dse_infer import dse_infer, encode_distribution


class DistributionMemo:
    def __init__(self, capacity=16384):
        self.capacity=capacity;self.cache=OrderedDict();self.hits=0;self.misses=0

    def __call__(self, values):
        values=np.asarray(values)
        key=(values.dtype.str,values.shape,values.tobytes(order='C'))
        if key in self.cache:
            self.hits+=1;self.cache.move_to_end(key);return self.cache[key]
        self.misses+=1;out=encode_distribution(values)
        out.setflags(write=False)
        self.cache[key]=out
        if len(self.cache)>self.capacity:self.cache.popitem(last=False)
        return out


class CachedInputs(Inputs):
    def __init__(self, bench, l1):
        super().__init__(bench,l1)
        self.distributions=DistributionMemo()
        namespace=dict(dse_infer.__globals__,encode_distribution=self.distributions)
        self.encoder=FunctionType(dse_infer.__code__,namespace,dse_infer.__name__,
                                  dse_infer.__defaults__,dse_infer.__closure__)
        self.build_seconds=0.;self.build_count=0

    def build(self, c):
        assert c['l1_size']==self.l1
        start=time.perf_counter()
        with np.errstate(divide='ignore',invalid='ignore'):
            raw=np.stack([self.encoder(p,c) for p in self.pre])
        x=torch.from_numpy(self.norm.transform(raw)).float()
        assert x.shape==(50,4681) and torch.isfinite(x).all()
        self.build_count+=1;self.build_seconds+=time.perf_counter()-start
        return x
