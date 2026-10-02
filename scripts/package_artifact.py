"""Package current source and recorded evidence; never include runtime caches."""
import argparse
import hashlib
from pathlib import Path
import shutil
import zipfile

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('output')
parser.add_argument('--mirror', help='Also mirror the authored source into this explicit directory')
args = parser.parse_args()
source = []
for folder in ('chrysalis', 'cpp', 'configs', 'scripts', 'tests', 'docs', '.github'):
    source.extend(p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts
                  and p.suffix not in ('.pyd', '.so', '.pyc', '.pdb', '.lib'))
source.extend(root/name for name in ('README.md', 'CMakeLists.txt', 'setup.py', 'pyproject.toml', 'MANIFEST.in', 'CITATION.cff',
                                   'requirements.txt', 'requirements-windows-build.txt', 'requirements-datasets.txt', '.gitignore', '.gitattributes'))
if (root/'LICENSE').exists():
    source.append(root/'LICENSE')
files = list(source)
files.extend(p for p in (root/'data/smoke').rglob('*') if p.is_file())
run = root/'runs/smoke-final'
files.extend(p for p in run.iterdir() if p.is_file())
for folder in ['executed-sources', 'tables', 'durable-evidence', 'oracle', 'dual', 'base', 'seed-snapshot'] + [f'SeedAsync-{s}' for s in range(1, 6)]:
    files.extend(p for p in (run/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
files = sorted(set(files))
digests = [(hashlib.sha256(p.read_bytes()).hexdigest(), p.relative_to(root).as_posix()) for p in files]
manifest = ''.join(f'{digest}  {name}\n' for digest, name in digests)
output = Path(args.output).resolve()
output.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for p in files:
        archive.write(p, 'Chrysalis-reconstruction/' + p.relative_to(root).as_posix())
    archive.writestr('Chrysalis-reconstruction/ARTIFACT-SHA256.txt', manifest)
if args.mirror:
    dest = Path(args.mirror).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    for p in source:
        target = dest/p.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    # Mirror data and measurements as well as code, so table regeneration works
    # from the supplied workspace without first unpacking the ZIP.
    for p in files:
        if p in source:
            continue
        target = dest/p.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    (dest/'ARTIFACT-SHA256.txt').write_text(manifest)
    shutil.copyfile(output, dest/output.name)
print('files:', len(files), 'archive_bytes:', output.stat().st_size)
print('archive_sha256:', hashlib.sha256(output.read_bytes()).hexdigest())
