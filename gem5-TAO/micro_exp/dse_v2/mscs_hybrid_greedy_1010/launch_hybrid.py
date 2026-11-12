"""Bounded supervisor in a managed execution session."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

here=Path(__file__).resolve().parent; start=time.time()
with (here/'launch.log').open('x') as log:
    p=subprocess.Popen([sys.executable,'-u',str(here/'run_hybrid.py')],cwd=here,
                       stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (here/'process.json').write_text(json.dumps({'pid':p.pid,'started_unix':start,'timeout_seconds':1200},indent=2))
    try:
        code=p.wait(timeout=1200)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGTERM)
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL);p.wait()
        code=124
(here/'process_exit.json').write_text(json.dumps({'returncode':code,'elapsed_seconds':time.time()-start},indent=2))
print(json.dumps({'returncode':code,'elapsed_seconds':time.time()-start}),flush=True)
sys.exit(code)
