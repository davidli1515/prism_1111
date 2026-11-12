#!/usr/bin/env python3
"""Counterfactual: run_dse_v3_single.py with DECOUPLE disabled (P3['eps']=0 -> never a
plateau), i.e. MSCS = PROTECT only. Everything else identical. Usage: same CLI as
run_dse_v3_single.py (pass --out to a NEW dir)."""
import sys, runpy, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refine_v3
refine_v3.P3["eps"] = 0.0          # same dict object used as default arg -> takes effect
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_dse_v3_single.py"),
               run_name="__main__")
