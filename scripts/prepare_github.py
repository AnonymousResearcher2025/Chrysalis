"""Create a source-only GitHub tree with genuine recorded validation evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('destination')
parser.add_argument('--anonymous', action='store_true', help='redact identifying metadata in the public copy')
args = parser.parse_args()
dest = Path(args.destination).resolve()
if dest.exists() and any(dest.iterdir()):
    raise FileExistsError('release directory must be empty')
dest.mkdir(parents=True, exist_ok=True)

def copy(source, name=None):
    target = dest/(name or source.relative_to(root))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)

for directory in ('chrysalis', 'cpp', 'configs', 'scripts', 'tests', 'docs', '.github'):
    for source in (root/directory).rglob('*'):
        if source.is_file() and '__pycache__' not in source.parts and source.suffix not in ('.pyd', '.so', '.lib', '.pdb', '.pyc'):
            if args.anonymous and source.name == 'source-paper.pdf':
                continue
            copy(source)
for name in ('README.md', 'CMakeLists.txt', 'setup.py', 'pyproject.toml', 'MANIFEST.in', 'CITATION.cff',
             'requirements.txt', 'requirements-windows-build.txt', 'requirements-datasets.txt', '.gitignore', '.gitattributes'):
    copy(root/name)
if (root/'LICENSE').exists():
    copy(root/'LICENSE')
run = root/('runs/smoke-final' if (root/'runs/smoke-final').exists() else 'evidence/smoke')
for source in run.iterdir():
    if source.suffix in ('.json', '.jsonl'):
        copy(source, Path('evidence/smoke')/source.name)
for directory in ('tables', 'executed-sources', 'durable-evidence'):
    for source in (run/directory).rglob('*'):
        if source.is_file():
            copy(source, Path('evidence/smoke')/source.relative_to(run))
copy(root/'data/smoke/manifest.json' if (root/'data/smoke/manifest.json').exists() else run/'input-manifest.json',
     Path('evidence/smoke/input-manifest.json'))
copy(root/'data/release-check/download-manifest.json' if (root/'data/release-check/download-manifest.json').exists() else run/'download-manifest.json',
     Path('evidence/smoke/download-manifest.json'))
(dest/'evidence/README.md').write_text(
    '# Recorded validation evidence\n\n'
    'smoke contains genuine measured query records, input/configuration hashes, '
    'durable ledger exports, and their executed source snapshots. These are the '
    'reconstruction CPU smoke results, not the historical paper measurements. '
    'Large raw corpora, model weights and serving databases are outside the GitHub '
    'tree; the full companion archive includes the smoke serving databases.\n\n'
    'Run `python -m chrysalis.cli tables evidence/smoke` to regenerate tables. '
    'See docs/ARTIFACT.md for the original-experiment/code-release relationship.\n')
if args.anonymous:
    # The public helper contains no private identifiers. Derive names locally
    # from the source citation; private copies and recorded originals stay intact.
    citation = (root/'CITATION.cff').read_text()
    names = re.findall(r'(?:family-names|given-names):\s*([^\n]+)', citation)
    tokens = {n.strip().strip('"\'') for n in names}
    tokens |= {part for n in tokens for part in n.split('-') if len(part) >= 4}
    for i, part in enumerate(root.parts[:-1]):
        if part.lower() == 'users':
            tokens.add(root.parts[i + 1])
    email = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
    def redact_text(text):
        text = email.sub('[redacted-email]', text)
        for token in sorted(tokens, key=len, reverse=True):
            text = re.sub(r'(?<!\w)' + re.escape(token) + r'(?!\w)', '[redacted-name]', text, flags=re.I)
        return re.sub(r'hostname="[^"]*"', 'hostname="redacted-for-review"', text)
    def redact_json(value, key=None):
        if key == 'hostname':
            return 'redacted-for-review'
        if isinstance(value, dict):
            return {k: redact_json(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [redact_json(v) for v in value]
        if isinstance(value, str):
            if re.match(r'^[A-Za-z]:[\\/]', value):
                return '[redacted-local-path]/' + value.replace('\\', '/').rsplit('/', 1)[-1]
            return redact_text(value)
        return value
    changed = []
    for path in sorted(p for p in dest.rglob('*') if p.is_file()):
        # Never run metadata redaction over source code: an XML-regex literal
        # inside this helper is code, not a recorded hostname.
        if path.suffix not in ('.json', '.jsonl', '.md', '.cff', '.xml', '.txt', '.yml', '.toml', '.in'):
            continue
        original = path.read_text(encoding='utf-8-sig')
        if path.suffix == '.json':
            value = json.loads(original)
            redacted = redact_json(value)
            updated = json.dumps(redacted, indent=2) + '\n' if value != redacted else original
        elif path.suffix == '.jsonl':
            updated = ''.join(json.dumps(redact_json(json.loads(line))) + '\n' for line in original.splitlines() if line.strip())
            # Keep byte-identical records when no identifying fields were found.
            if all(json.loads(a) == json.loads(b) for a, b in zip(original.splitlines(), updated.splitlines())):
                updated = original
        else:
            updated = redact_text(original)
        if path.name == 'CITATION.cff':
            updated = re.sub(r'authors:\n(?:  .+\n)+', 'authors:\n  - name: Anonymous contributors\n', updated)
        if path.name == 'SPECIFICATION.md':
            updated = updated.replace('The delivered `docs/source-paper.pdf` is a checksum-identical copy for review.',
                                      'The source PDF is retained privately; the public copy records its checksum.')
        if path.name == 'RELEASE_READINESS.md':
            updated = updated.replace('CITATION.cff credits the authors listed on the supplied paper in this full copy.',
                                      'CITATION.cff uses anonymous contributors for this public review copy.')
            updated = updated.replace('Anonymous\nreview requires an explicit public-copy redaction pass; private originals remain\nthe authoritative evidence.',
                                      'This public copy has been anonymized as described in ANONYMIZATION.md;\nprivate originals remain the authoritative evidence.')
        if path.name == 'README.md':
            updated = updated.replace('The archive includes `ARTIFACT-SHA256.txt`. Verify all packaged bytes with\n`python scripts/verify_artifact.py /path/to/Chrysalis-reconstruction.zip`.',
                                      'The public archive includes `RELEASE-SHA256.txt`. Verify all packaged bytes\nwith `python scripts/verify_artifact.py /path/to/Chrysalis-GitHub-source.zip`.\nSee [anonymization](docs/ANONYMIZATION.md) for the public redaction record. The full\ncompanion archive is private and must not be uploaded during anonymous review.')
        if updated != original:
            path.write_text(updated, encoding='utf-8')
            changed.append(path.relative_to(dest).as_posix())
    (dest/'docs/ANONYMIZATION.md').write_text(
        '# Anonymous public release\n\n'
        'This copy omits the author-bearing source PDF and uses anonymous citation '
        'metadata. Hostnames, personal paths, names and emails in validation metadata '
        'are explicitly redacted. Private original evidence is preserved separately. '
        'Public redacted files are not claimed to be byte-identical to private logs. '
        'Measured numeric values, timestamps, query IDs, scientific configurations '
        'and algorithm code are unchanged. RELEASE-SHA256.txt hashes this public copy.\n')
    (dest/'REDACTION.json').write_text(json.dumps(dict(anonymous=True, omitted=['docs/source-paper.pdf'],
         redacted_files=changed, numeric_measurements_changed=False, private_originals_preserved=True), indent=2))
paths = sorted(p for p in dest.rglob('*') if p.is_file())
manifest = ''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(dest).as_posix()}\n' for p in paths)
(dest/'RELEASE-SHA256.txt').write_text(manifest)
archive = dest.parent/'Chrysalis-GitHub-source.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as out:
    for p in sorted(p for p in dest.rglob('*') if p.is_file()):
        out.write(p, 'Chrysalis/' + p.relative_to(dest).as_posix())
print(json.dumps(dict(directory=str(dest), files=len(paths), archive=str(archive),
                      archive_bytes=archive.stat().st_size,
                      archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                      license_selected=(dest/'LICENSE').exists()), indent=2))
