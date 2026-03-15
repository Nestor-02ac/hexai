"""
Build script for Cython extensions (chex_board + cmcts_hex).

Usage:
  python setup.py build_ext --inplace

This compiles:
  - chex_board.pyx -> chex_board.*.so  (Cython Hex board with Union-Find)
  - cmcts_hex.pyx  -> cmcts_hex.*.so   (Cython MCTS with UCT + RAVE)

Compiler directives disable bounds/wrap/none checks for maximum speed.
"""
from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

extensions = [
    Extension(
        "chex_board",
        ["chex_board.pyx"],
    ),
    Extension(
        "cmcts_hex",
        ["cmcts_hex.pyx"],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'nonecheck': False,
            'language_level': '3',
        },
    ),
)
