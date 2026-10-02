import numpy as np
import pytest
from tests.test_system import make_index


def test_legacy_candidate_claim_does_not_rotate_corpus(tmp_path):
    idx, _, exact = make_index(tmp_path)
    c = idx.claim(79, 'boundary')
    assert c
    assert sum(n['state'] == 'legacy' for n in idx.store.nodes()) == 79
    assert np.allclose(idx.graph.vector(79), exact[79])
    idx.rotate()
    assert np.allclose([idx.graph.vector(i) for i in range(80)], exact)
    assert idx.publish_native(c, exact[79], 'query')
    with pytest.raises(RuntimeError, match='chained'):
        idx.configure('third-version', [])
    idx.close()
