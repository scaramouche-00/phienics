import setuptools
import re

VERSIONFILE = "_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setuptools.setup(
    name="phienics",
    version=verstr,
    author="Daniela Saadeh",
    author_email="daniela.saadeh@gmail.com",
    description="Screening with the finite element method",
    long_description_content_type="text/markdown",
    url="https://github.com/scaramouche-00/phienics",
    packages=setuptools.find_packages(),
)
