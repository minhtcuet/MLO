from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("cython_code", ["cython_code.pyx"], include_dirs=[np.get_include()])
]

setup(ext_modules=cythonize(extensions))

# python setup.py build_ext --inplace
