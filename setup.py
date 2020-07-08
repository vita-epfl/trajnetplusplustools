from setuptools import setup

# extract version from __init__.py
with open('trajnetplusplustools/__init__.py', 'r') as f:
    VERSION_LINE = [l for l in f if l.startswith('__version__')][0]
    VERSION = VERSION_LINE.split('=')[1].strip()[1:-1]


setup(
    name='trajnetplusplustools',
    version=VERSION,
    packages=[
        'trajnetplusplustools',
    ],
    license='MIT',
    description='Trajnet++ tools.',
    long_description=open('README.rst').read(),
    author='Parth Kothari',
    author_email='parth.kothari@epfl.ch',
    url='https://github.com/vita-epfl/trajnetplusplustools',

    install_requires=[
        'numpy',
        'scipy',
        'pykalman',
    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
        ],
        'plot': [
            'matplotlib',
        ]
    },

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: PyPy',
    ]
)
