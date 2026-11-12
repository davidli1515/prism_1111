"""Bounded foreground supervisor; output persists while the tool session yields."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

here = Path(__file__).resolve().parent
start = time.time()
with (here/'launch.log').open('x') as log:
    process = subprocess.Popen([sys.executable, '-u', str(here/'run.py')], cwd=here,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (here/'process.json').write_text(json.dumps({'pid': process.pid, 'started_unix': start,
                                                'timeout_seconds': 1800}, indent=2))
    try:
        code = process.wait(timeout=1800)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        code = 124
(here/'process_exit.json').write_text(json.dumps({'returncode': code,
                                                  'elapsed_seconds': time.time()-start}, indent=2))
print(json.dumps({'returncode': code, 'elapsed_seconds': time.time()-start}), flush=True)
sys.exit(code)
