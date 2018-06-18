from setuptools import setup

# extract version from __init__.py
with open('trajnettools/__init__.py', 'r') as f:
    version_line = [l for l in f if l.startswith('__version__')][0]
    VERSION = version_line.split('=')[1].strip()[1:-1]


setup(
    name='trajnettools',
    version=VERSION,
    packages=[
        'trajnettools',
    ],
    license='MIT',
    description='Trajnet tools.',
    long_description=open('README.rst').read(),
    author='Sven Kreiss',
    author_email='me@svenkreiss.com',
    url='https://github.com/svenkreiss/trajnettools',

    install_requires=[
        'numpy',
        'pysparkling',
        'scipy',
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
    entry_points={
        'console_scripts': [],
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
