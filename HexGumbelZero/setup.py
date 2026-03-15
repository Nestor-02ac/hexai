"""
Build Cython board extension.
Only chex_board needs compilation; MCTS is pure Python (needs PyTorch callbacks).

Usage: python setup.py build_ext --inplace
"""

from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

extensions = [
    Extension("chex_board", ["chex_board.pyx"]),
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
