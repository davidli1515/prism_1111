"""Bounded separate processes enforce prediction/label phase separation."""
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

here = Path(__file__).resolve().parent; start = time.perf_counter(); phases = []
for name in ('prepare_disagreement','screen_disagreement','evaluate_disagreement'):
    begin = time.perf_counter()
    with (here/f'{name}.log').open('x') as log:
        child = subprocess.Popen([sys.executable,'-u',str(here/f'{name}.py')],
            cwd=here, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=600)
        except subprocess.TimeoutExpired:
            subprocess.run(['kill','-TERM','--',f'-{child.pid}'], check=False)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                subprocess.run(['kill','-KILL','--',f'-{child.pid}'], check=False); child.wait()
            code = 124
    phases.append(dict(phase=name,returncode=code,wall_seconds=time.perf_counter()-begin))
    print(json.dumps(phases[-1]),flush=True)
    if code:
        break
with (here/'launch_exit.json').open('x') as f:
    json.dump(dict(returncode=code,phases=phases,wall_seconds=time.perf_counter()-start),f,indent=2)
sys.exit(code)
