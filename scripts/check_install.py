"""Run outside the checkout after installing the wheel; no model download."""
import json
from pathlib import Path
import chrysalis
from chrysalis import _core
from chrysalis.settings import load

model = load('configs/models.lock.json')
assert len(model['mpnet']['revision']) == 40
assert load('configs/smoke.json')['R'] == 1
graph = _core.Graph(4, 16, 1.2, 42)
graph.build([[1., 0.], [0., 1.], [-1., 0.]], [0, 0, 0])
assert graph.search([1., 0.], 4)[0].id == 0
assert graph.check_reverse()
print(json.dumps(dict(package_file=str(Path(chrysalis.__file__).resolve()),
                      kernel_file=str(Path(_core.__file__).resolve()),
                      version=chrysalis.__version__, bundled_configs=True,
                      native_search=True), indent=2))
