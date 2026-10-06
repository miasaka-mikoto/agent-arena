"""Run the real regression suite without external service access."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

GUARD = """import ipaddress, socket
_connect = socket.socket.connect
def offline_connect(self, address):
    host = address[0] if isinstance(address, tuple) else None
    if host is not None and host not in ('localhost', '127.0.0.1', '::1'):
        try:
            local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = False
        if not local:
            raise RuntimeError('OFFLINE_TESTS: external connections are disabled')
    return _connect(self, address)
socket.socket.connect = offline_connect
"""

def main() -> int:
    with tempfile.TemporaryDirectory(prefix='offline-tests-') as temp:
        root = Path(temp)
        (root / 'sitecustomize.py').write_text(GUARD, encoding='utf-8')
        env = os.environ.copy()
        for key in list(env):
            if any(word in key.upper() for word in ('API_KEY', 'SECRET', 'TOKEN', 'PASSWORD')):
                env.pop(key, None)
        env['PYTHONPATH'] = str(root) + os.pathsep + env.get('PYTHONPATH', '')
        report = root / 'results.xml'
        completed = subprocess.run([sys.executable, '-m', 'pytest', '-q', '--junitxml=' + str(report)], env=env)
        if completed.returncode != 0:
            return completed.returncode
        if not report.is_file():
            raise RuntimeError('Test runner produced no JUnit report')
        xml = ET.parse(report).getroot()
        suites = [xml] if xml.tag == 'testsuite' else list(xml.findall('testsuite'))
        counts = {key: sum(int(suite.get(key, '0')) for suite in suites) for key in ('tests', 'failures', 'errors', 'skipped')}
        passed = counts['tests'] - counts['failures'] - counts['errors'] - counts['skipped']
        print('OFFLINE_TEST_COUNTS', counts, 'passed=', passed, flush=True)
        if passed <= 0 or counts['failures'] or counts['errors']:
            raise RuntimeError('No passing tests, or a failing JUnit result')
        summary = os.environ.get('GITHUB_STEP_SUMMARY')
        if summary:
            with Path(summary).open('a', encoding='utf-8') as output:
                output.write(f"Tests: {counts['tests']}; passed: {passed}; skipped: {counts['skipped']}.\n")
                output.write('External connections blocked for Python test processes. This is not a GUI or human acceptance claim.\n')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
