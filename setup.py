from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import importlib.util
import os
from pathlib import Path
import subprocess
import sys


class PortableBuildExt(build_ext):
    def run(self):
        if os.name == 'nt' and importlib.util.find_spec('ziglang'):
            root = Path(__file__).resolve().parent
            for extension in self.extensions:
                subprocess.run([sys.executable, str(root/'scripts/build.py'), '--output',
                                str(Path(self.get_ext_fullpath(extension.name)).resolve())], check=True)
        else:
            super().run()


setup(ext_modules=[Pybind11Extension('chrysalis._core', ['cpp/core.cpp'], cxx_std=17)],
      cmdclass={'build_ext': PortableBuildExt})
