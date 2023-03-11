from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension("micronlp",
                      ["setup.cpp"],
                      define_macros=[('VERSION_INFO', __version__)],
                      )
]

setup(
    name="micronlp",
    version=__version__,
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False
)
