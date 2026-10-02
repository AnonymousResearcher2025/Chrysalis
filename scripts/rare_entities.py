"""Evaluate supplied entity annotations against measured query outputs."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chrysalis.datasets import read_jsonl, rare_entities, sha256
from chrysalis.evaluation import summary
p = argparse.ArgumentParser()
p.add_argument('--corpus-annotations', required=True)
p.add_argument('--query-annotations', required=True)
p.add_argument('--annotation-method', required=True, help='extractor/revision, casing, aliases and counting scope')
p.add_argument('--run', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
ids, frequency = rare_entities(read_jsonl(a.corpus_annotations), read_jsonl(a.query_annotations))
groups = {}
for path in Path(a.run).glob('queries-*.jsonl'):
    for row in read_jsonl(path):
        if row['query_id'] in ids:
            groups.setdefault(row['label'], []).append(row)
result = dict(annotation_method=a.annotation_method, corpus_hash=sha256(a.corpus_annotations),
              query_hash=sha256(a.query_annotations), counting_rule='document_frequency_bottom_decile_with_ties',
              rare_query_ids=ids, entity_frequencies=frequency,
              measured={name:summary(rows) for name, rows in groups.items()})
Path(a.output).write_text(json.dumps(result, indent=2))
