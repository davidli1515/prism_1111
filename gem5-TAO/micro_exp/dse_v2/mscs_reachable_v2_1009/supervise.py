#!/usr/bin/env python3
"""One bounded background job with an explicit final process outcome."""
import json
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
OUT=HERE/'results_s42_01'


def main():
    lock=(HERE/'supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if OUT.exists() or (HERE/'supervisor_status.json').exists():
        raise FileExistsError('Refusing to overwrite an existing run')
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    record={'supervisor_pid':os.getpid(),'started_unix':time.time(),'timeout_seconds':600}
    def persist():
        p=HERE/'supervisor_status.tmp';p.write_text(json.dumps(record,indent=2));p.replace(HERE/'supervisor_status.json')
    with (HERE/'run_s42_01.log').open('x') as log:
        p=subprocess.Popen([sys.executable,'-u',str(HERE/'run.py')],stdout=log,stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL,start_new_session=True,
                           env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
        record.update(child_pid=p.pid,phase='running');persist()
        try:
            rc=p.wait(timeout=600);record.update(returncode=rc,phase='finished' if rc==0 else 'failed')
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            record.update(phase='timeout',returncode=p.returncode)
    record['finished_unix']=time.time();persist()


if __name__=='__main__':main()
