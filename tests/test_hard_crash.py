import subprocess
import sys
import time
import numpy as np
import pytest
from chrysalis.index import Index
from tests.test_system import make_index


@pytest.mark.parametrize('stage', ['data_sync', 'manifest_sync', 'before_reclaim'])
@pytest.mark.parametrize('operation', ['rotation', 'native'])
def test_real_process_exit_recovery(tmp_path, stage, operation):
    idx, _, exact = make_index(tmp_path)
    if operation == 'native':
        idx.rotate()
        if stage == 'before_reclaim':
            # Resolve the rest of first segment so node 0's publication really
            # supersedes the segment, making reclamation reachable.
            for i in range(1, 11):
                c = idx.claim(i, 'prepare', 60)
                idx.publish_native(c, exact[i], 'query')
    idx.close()
    child = subprocess.run([sys.executable, '-m', 'tests.crash_process', str(tmp_path), stage, operation], timeout=60)
    assert child.returncode == 17
    time.sleep(.02)
    idx = Index(tmp_path)
    # Simulate lease expiry only after verifying real WAL recovery; a tiny
    # real-time lease makes the crash barrier flaky under CPU load.
    idx.store.recover(now=time.time() + 60)
    idx.reload()
    idx.rotate()
    if operation == 'native':
        n = idx.store.node(0)
        assert n['state'] == ('bridged' if stage == 'data_sync' else 'native')
        assert idx.store.get('meta')['flips'] == (0 if stage == 'data_sync' else 11 if stage == 'before_reclaim' else 1)
    assert np.allclose([idx.graph.vector(i) for i in range(80)], exact)
    assert {p.relative_to(tmp_path).as_posix() for p in (tmp_path / 'vectors').glob('*.npy')} == {n['file'] for n in idx.store.nodes()}
    idx.close()
