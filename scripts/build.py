"""Portable C++17 extension build; CMake is also supported."""
import os
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import pybind11

root = Path(__file__).resolve().parents[1]
out = root / 'chrysalis' / ('_core' + sysconfig.get_config_var('EXT_SUFFIX'))
parser = argparse.ArgumentParser()
parser.add_argument('--output')
args = parser.parse_args()
if args.output:
    out = Path(args.output).resolve()
out.parent.mkdir(parents=True, exist_ok=True)
includes = [pybind11.get_include(), sysconfig.get_path('include')]
if os.name == 'nt':
    if os.environ.get('ZIG_EXECUTABLE'):
        zig = Path(os.environ['ZIG_EXECUTABLE'])
    else:
        import ziglang
        zig = Path(ziglang.__file__).parent / 'zig.exe'
    base = Path(sys.base_prefix)
    cmd = [str(zig), 'c++', '-std=c++17', '-O2', '-shared', '-DNDEBUG',
           str(root / 'cpp/core.cpp'), '-o', str(out),
           str(base / 'libs' / f'python{sys.version_info.major}{sys.version_info.minor}.lib')]
else:
    cmd = [shutil.which('c++') or 'c++', '-std=c++17', '-O2', '-shared', '-fPIC', '-DNDEBUG',
           str(root / 'cpp/core.cpp'), '-o', str(out)]
cmd += ['-I' + x for x in includes]
subprocess.run(cmd, check=True, cwd=root)
print(out)
