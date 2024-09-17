from setuptools import setup, find_packages
from os import path

DIR = path.dirname(path.abspath(__file__))

DESCRIPTION = "A toolbox of common matching methods"

AUTHORS = 'Trouvaille98'

URL = 'https://github.com/Trouvaille98/pymatchingtools'

EMAIL = 'dulingzhi.0710@gmail.com'

with open(path.join(DIR, 'requirements.txt')) as f:
    INSTALL_PACKAGES = f.read().splitlines()

with open(path.join(DIR, 'README.md')) as f:
    README = f.read()

VERSION = '0.1.0'

setup(
    name='pymatchingtools',
    packages=find_packages(),
    description=DESCRIPTION,
    long_description=README,
    long_description_content_type='text/markdown',
    license="MIT",
    install_requires=INSTALL_PACKAGES,
    version=VERSION,
    url=URL,
    author=AUTHORS,
    author_email=EMAIL,
    keywords=['causal inference', 'PSM', 'Matching', 'observational study', 'pymatchingtools', 'psm', 'propensity score', 'propensity score matching', 'balance check'],
    python_requires='>=3.7',
    zip_safe=True
)