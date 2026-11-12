#!/usr/bin/env python3
"""Bounded background sequence: tests -> experiment -> independent validation."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    lock = (HERE/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (HERE/'supervisor_status.json').exists() or (HERE/'results_s42_01').exists():
        raise FileExistsError('Refusing to overwrite a previous run')
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    start = time.monotonic()
    record = {'supervisor_pid': os.getpid(), 'started_unix': time.time(), 'timeout_seconds': 1800}
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    def persist():
        temp = HERE/'supervisor_status.tmp'
        temp.write_text(json.dumps(record, indent=2)); temp.replace(HERE/'supervisor_status.json')
    tasks = [('tests', [sys.executable, '-m', 'unittest', '-v', 'test_intent']),
             ('experiment', [sys.executable, '-u', str(HERE/'run.py')]),
             ('independent_validation', [sys.executable, '-u', str(HERE/'validate_result.py')])]
    for name, command in tasks:
        with (HERE/f'{name}.log').open('x') as log:
            child = subprocess.Popen(command, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
            record.update(phase=name, child_pid=child.pid); persist()
            try:
                rc = child.wait(timeout=max(1, 1800-(time.monotonic()-start)))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
                record.update(phase='timeout', failed_stage=name, returncode=child.returncode,
                              finished_unix=time.time()); persist(); return 1
        record.setdefault('completed', []).append({'stage': name, 'returncode': rc})
        if rc:
            record.update(phase='failed', failed_stage=name, returncode=rc, finished_unix=time.time())
            persist(); return rc
    record.update(phase='complete', returncode=0, finished_unix=time.time())
    persist()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
