"""Child-process hard exit injected at publication barriers."""
import os
import sys
from chrysalis.index import Index
index = Index(sys.argv[1])
def crash(stage):
    if stage == sys.argv[2]:
        os._exit(17)
index.store.crash = crash
if sys.argv[3] == 'rotation':
    index.rotate_one()
else:
    claim = index.claim(0, 'crash-worker', 30)
    index.publish_native(claim, index.graph.vector(0), 'query')
raise RuntimeError('crash point not reached')
