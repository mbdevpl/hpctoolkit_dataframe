"""Setup script for hpctoolkit_dataframe package."""

import boilerplates.setup


class Package(boilerplates.setup.Package):
    """Package metadata."""

    name = 'hpctoolkit_dataframe'
    description = 'Operate on HPCtoolkit XML database files as pandas DataFrames.'
    url = 'https://github.com/mbdevpl/hpctoolkit_dataframe'
    classifiers = [
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Utilities']
    keywords = ['hpc', 'high-performance computing', 'performance', 'profiling']


if __name__ == '__main__':
    Package.setup()
