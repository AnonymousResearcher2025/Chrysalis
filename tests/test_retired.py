import numpy as np
from chrysalis import _core
from tests.test_system import make_index


def test_retirement_bypasses_interval_and_heat_machinery(tmp_path, monkeypatch):
    idx, _, exact = make_index(tmp_path)
    idx.rotate()
    for i in range(80):
        c = idx.claim(i, 'converge')
        idx.publish_native(c, exact[i], 'scheduled')
    idx.retire()
    def forbidden(*args):
        raise AssertionError('retired serving invoked interval machinery')
    monkeypatch.setattr(_core, 'certificate', forbidden)
    monkeypatch.setattr(_core, 'ambiguity', forbidden)
    before = {key:idx.store.get(key) for key in idx.store.keys('heat/')}
    result = idx.search(exact[0])
    after = {key:idx.store.get(key) for key in idx.store.keys('heat/')}
    assert before == after
    assert result['bound'] is None and result['requested'] == 0
    assert result['results'][0] == 0
    idx.close()
