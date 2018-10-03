"""Setup script for hpctoolkit_dataframe package."""

import setup_boilerplate


class Package(setup_boilerplate.Package):

    """Package metadata."""

    name = 'hpctoolkit_dataframe'
    description = 'operate on HPCtoolkit XML database files as pandas DataFrames'
    download_url = 'https://github.com/mbdevpl/hpctoolkit_dataframe'
    classifiers = [
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Utilities']
    keywords = ['hpc', 'high-performance computing', 'performance', 'profiling']


if __name__ == '__main__':
    Package.setup()
