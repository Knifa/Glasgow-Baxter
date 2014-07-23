from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['glasgow_baxter_helpers'],
    package_dir={'': 'src/'}
)

setup(**d)