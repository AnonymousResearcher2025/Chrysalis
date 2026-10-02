"""Verify a packaged artifact and optionally extract into an empty directory."""
import argparse
import hashlib
from pathlib import Path
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument('archive')
parser.add_argument('--extract')
args = parser.parse_args()
with zipfile.ZipFile(args.archive) as archive:
    prefix = 'Chrysalis-reconstruction/'
    manifest_name = 'ARTIFACT-SHA256.txt'
    if prefix + manifest_name not in archive.namelist():
        prefix, manifest_name = 'Chrysalis/', 'RELEASE-SHA256.txt'
    rows = archive.read(prefix + manifest_name).decode().splitlines()
    for row in rows:
        expected, name = row.split('  ', 1)
        assert hashlib.sha256(archive.read(prefix + name)).hexdigest() == expected, name
    if args.extract:
        dest = Path(args.extract).resolve()
        if dest.exists() and any(dest.iterdir()):
            raise FileExistsError('extract destination must be empty')
        for name in archive.namelist():
            target = (dest / name).resolve()
            if not target.is_relative_to(dest):
                raise ValueError('archive path escapes destination')
        archive.extractall(dest)
print('verified files:', len(rows))
