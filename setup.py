#!/usr/bin/env python

import os
from distutils.core import setup

import versioneer

# Utility function to read the README file.
def read(fname):
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
                        'matplotlib>=3.4' 
                        ],
      extras_require={'test': ['pytest'],
                      'docs': ['sphinx>=4.0']
                      },
      python_requires='>=3.7',
      )
