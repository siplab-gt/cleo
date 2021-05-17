#!/usr/bin/env python

import os
from setuptools import setup

import versioneer


def read(fname):
    """Utility function to read the README file."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='clocsim',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Closed-loop experiment simulator and testbed',
      author='Kyle Johnsen',
      author_email='kjohnsen@gatech.edu',
      url='siplab.gatech.edu',
      packages=['clocsim'],
      long_description=read('README.md'),
      license='MIT',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
      ],
      provides='clocsim',
      keywords='Brian simulator closed loop neuroscience electrode optogenetics',
      install_requires=['brian2>=2.4',
                        'matplotlib>=3.4',
                        'nptyping',
                        ],
      extras_require={'test': ['pytest'],
                      'docs': ['sphinx>=4.0']
                      },
      setup_requires=['pytest-runner'],
      tests_require=['pytest', 'flake8', 'pytest-flake8', 'pep8-naming',
                     'flake8-docstrings', 'flake8-pytest-style'],
      python_requires='>=3.7',
      )
