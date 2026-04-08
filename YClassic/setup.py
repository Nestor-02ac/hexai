"""
Build script for YClassic Cython extensions.

Usage:
  python setup.py build_ext --inplace
"""

from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension


extensions = [
    Extension(
        "cy_board",
        ["cy_board.pyx"],
    ),
    Extension(
        "cmcts_y",
        ["cmcts_y.pyx"],
    ),
]


setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
            "language_level": "3",
        },
    ),
)
