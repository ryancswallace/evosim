import os
from setuptools import setup, find_packages

root = os.path.abspath(os.path.dirname(__file__))
try:
    long_desc = open(os.path.join(root, 'README.md')).read()
except Exception:
    long_desc = '<Missing README.md>'
    print('Missing README.md')

setup(
    name='evosim',
    version='0.1',
    description='Evolution simulator',
    long_description=long_desc,
    author='Ryan Wallace',
    author_email='ryanwallace@college.harvard.edu',
    url='https://github.com/ryanwallace96/evosim',
    license='MIT License',
    packages=find_packages(),
    install_requires=['pandas', 'numpy'],
    include_package_data=True,
    keywords=('evolution', 'simulation', 'simulator'),
    classifiers=[  
        #https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 1 - Planning',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
    ]
)