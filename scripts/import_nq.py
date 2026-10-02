"""NQ query import; supplied release/split must be recorded in prepare manifest."""
import argparse
import gzip
import json
p = argparse.ArgumentParser()
p.add_argument('--input', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
with gzip.open(a.input, 'rt', encoding='utf-8') as source, open(a.output, 'w', encoding='utf-8') as dest:
    for line in source:
        item = json.loads(line)
        dest.write(json.dumps(dict(id=str(item['example_id']), raw=item['question_text'])) + '\n')
