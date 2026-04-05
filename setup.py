"""
setup.py – build the board2048_ext C extension.

Usage:
    python setup.py build_ext --inplace
"""

from setuptools import setup, Extension

module = Extension(
    "board2048_ext",
    sources=["board2048_ext.c"],
    extra_compile_args=["-std=c11"],
)

setup(
    name="board2048_ext",
    version="1.0",
    description="Python C extension: board2048",
    ext_modules=[module],
)
