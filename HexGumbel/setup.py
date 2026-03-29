"""Build Cython extensions for HexGumbel."""

import numpy as np
from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

extensions = [
    Extension("chex_board", ["chex_board.pyx"], include_dirs=[np.get_include()]),
    Extension("cgumbel_mcts", ["cgumbel_mcts.pyx"], include_dirs=[np.get_include()]),
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
