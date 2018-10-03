"""Tests for HPCtoolkitDataFrame."""

import contextlib
import datetime
import logging
import os
import pathlib
import sys
import unittest

import hpctoolkit_dataframe
from hpctoolkit_dataframe import HPCtoolkitDataFrame

_HERE = pathlib.Path(__file__).parent.resolve()

_LOG = logging.getLogger(__name__)

_LOGS = pathlib.Path(os.environ.get('LOGGING_PATH', str(_HERE.parent.joinpath(
    'results', datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))))

_LOGS.mkdir(parents=True)

logging.basicConfig(
    filename=_LOGS.joinpath('logging.log'),
    level=getattr(logging, os.environ.get('LOGGING_LEVEL', 'warning').upper(), logging.WARNING)
    )

_LOG.setLevel(logging.DEBUG)


class Tests(unittest.TestCase):

    def test_load(self):
        for path in _HERE.glob('data/experiment*.xml'):
            with self.subTest(path=path):
                df = HPCtoolkitDataFrame(path=path)
                hot_path = df.hot_path()
                self.assertIsInstance(hot_path, HPCtoolkitDataFrame)
                for field in getattr(HPCtoolkitDataFrame, '_metadata'):
                    self.assertEqual(getattr(df, field), getattr(hot_path, field))
                compact = df.compact
                self.assertIsInstance(compact, HPCtoolkitDataFrame)
                for field in getattr(HPCtoolkitDataFrame, '_metadata'):
                    self.assertEqual(getattr(df, field), getattr(compact, field))

    def test_load_shallow(self):
        for path in _HERE.glob('data/experiment*.xml'):
            with self.subTest(path=path):
                df = HPCtoolkitDataFrame(path=path, max_depth=4)
                self.assertFalse(df.at_depth(4).empty)
                self.assertTrue(df.at_depth(5).empty)

    def test_load_with_callsite(self):
        HPCtoolkitDataFrame._skip_callsite = False
        for path in _HERE.glob('data/experiment*.xml'):
            with self.subTest(path=path):
                df = HPCtoolkitDataFrame(path=path, max_depth=5)
                self.assertIsNotNone(df)
        HPCtoolkitDataFrame._skip_callsite = True

    @unittest.skipIf(sys.version_info[:2] != (3, 6), 'run only on Python 3.6')
    def test_performance(self):
        import line_profiler
        profiler = line_profiler.LineProfiler()
        profiler.add_module(hpctoolkit_dataframe)
        for path in _HERE.glob('data/experiment*.xml'):
            profiler.enable()
            df = HPCtoolkitDataFrame(path=path, max_depth=None)
            df.hot_path()
            profiler.disable()
        with _LOGS.joinpath('hpctoolkit_dataframe_profile.log').open('w') as profiler_log:
            with contextlib.redirect_stdout(profiler_log):
                profiler.print_stats()
