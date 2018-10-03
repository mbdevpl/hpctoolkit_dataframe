"""Operate on HPCtoolkit XML database files as pandas DataFrames."""

from cmath import sqrt  # used in formulas for metrics
import collections
import logging
import pathlib
import pprint
import re
import typing as t
import xml.etree.ElementTree as ET

import numpy as np
import ordered_set
import pandas as pd

_LOG = logging.getLogger(__name__)

_MEASUREMENT_TYPES = {
    'PF': 'procedure frame',
    'C': 'callsite',
    'Pr': 'procedure',
    'S': 'statement',
    'L': 'loop'}

_LOCATION_TYPES = {
    'lm': 'module',
    'f': 'file',
    'l': 'line',
    'n': 'procedure',
    'i': 'id',
    }

_LOCATION_COLUMNS = [
    'callpath', 'module path', 'module', 'file path', 'file', 'line', 'procedure', 'id',
    'type']

_COMPACT_LOCATION_COLUMNS = ['module', 'file', 'line', 'procedure', 'type']

_LOCATION_DATA_TRANSFORMERS = {
    ('lm', _LOCATION_TYPES['lm'] + ' path'): lambda self, data: self._modules_by_id[int(data)],
    ('lm', None): lambda self, data: self._modules_by_id[int(data)].name,
    ('f', _LOCATION_TYPES['f'] + ' path'): lambda self, data: self._files_by_id[int(data)],
    ('f', None): lambda self, data: self._files_by_id[int(data)].name,
    ('l', None): lambda _, data: int(data),
    ('n', None): lambda self, data: self._procedures_by_id[int(data)],
    ('i', None): lambda _, data: int(data)}

_ROOT_INDEX = -1

_NORMALIZATION_CONSTANT = 2 * np.pi


def _read_xml(path: pathlib.Path) -> ET.ElementTree:
    with path.open() as xml_file:
        xml_data = ET.parse(xml_file)
    return xml_data


def _metrics_formula_sub_predicate(match: t.Match) -> str:
    return 'data.get(self._metrics_by_id[{}])'.format(match.group()[1:])


def _derive_metrics_formulas(
        metrics: ET.Element) -> t.Dict[str, t.Tuple[str, t.Callable[[pd.DataFrame, dict], t.Any]]]:
    metrics_formulas = {}
    for metric in metrics:
        formulas = metric.findall('./MetricFormula')
        for formula in formulas:
            if formula.attrib['t'] != 'finalize':
                continue
            raw_formula = formula.attrib['frm']
            formula_code = re.sub('\$[0-9]+', _metrics_formula_sub_predicate, raw_formula)
            compiled_formula = eval('lambda self, data: {}'.format(formula_code), None, None)
            metrics_formulas[metric.attrib['n']] = (formula_code, compiled_formula)
            break
    return metrics_formulas


def _callpath_filter(series: pd.Series, fragments: t.Sequence[t.Sequence[int]],
                     prefix: t.Sequence[int], suffix: t.Sequence[int]) -> bool:
    callpath = series.at['callpath']
    if fragments:
        raise NotImplementedError(
            'filtering by arbitrary fragment "{}" not supported'.format(fragments))
    # for fragment in fragments:
    #    assert fragment, fragments
    #    # TODO: implement
    if prefix and (len(callpath) < len(prefix) or callpath[:len(prefix)] != prefix):
        return False
    if suffix and (len(callpath) < len(suffix) or callpath[-len(suffix):] != suffix):
        return False
    return True


def _str_or_regex_sequence_filter(
        series: pd.Series, column: str, fragments: t.Sequence[t.Sequence[t.Union[t.Pattern, str]]],
        prefix: t.Sequence[t.Union[t.Pattern, str]],
        suffix: t.Sequence[t.Union[t.Pattern, str]]) -> bool:
    value = series.at[column]
    if fragments:
        raise NotImplementedError(
            'filtering by arbitrary fragment "{}" not supported'.format(fragments))
    # for fragment in fragments:
    #    assert fragment, fragments
    #    # TODO: implement
    if prefix:
        if len(value) < len(prefix):
            return False
        for value_item, prefix_item in zip(value[:len(prefix)], prefix):
            if not (prefix_item.fullmatch(value_item) if isinstance(prefix_item, t.Pattern)
                    else prefix_item == value_item):
                return False
    if suffix:
        if len(value) < len(suffix):
            return False
        for value_item, suffix_item in zip(value[-len(suffix):], suffix):
            if not (suffix_item.fullmatch(value_item) if isinstance(suffix_item, t.Pattern)
                    else suffix_item == value_item):
                return False
    return True


def _depth_filter(
        series: pd.Series, min_depth: t.Optional[int], max_depth: t.Optional[int]) -> bool:
    depth = len(series.at['callpath'])
    if min_depth is not None and depth < min_depth or max_depth is not None and depth > max_depth:
        return False
    return True


class HPCtoolkitDataFrame(pd.DataFrame):

    """Extension of pandas DataFrame tailored to HPCtoolkit"""

    _metadata = ['_db_path', '_metrics_by_id', '_metrics_formulas', '_modules_by_id',
                 '_files_by_id', '_procedures_by_id', '_meaningful_columns']
    # , '_max_depth'

    _fundamental_column_prefix = 'CPUTIME (usec):'

    _percentage_column_candidates = ['Mean (I)', 'Sum (I)']

    _compact_column_suffixes = ['', ' ratio of total', ' ratio of parent']

    _hot_path_column_suffix = ' ratio of total'

    _skip_callsite = True
    """Skip over callsite nodes to avoid over-complicating the calltree."""

    @property
    def _constructor(self):
        return HPCtoolkitDataFrame

    def __init__(self, *args,
                 path: pathlib.Path = None, max_depth: t.Optional[int] = None, **kwargs):
        if path is None:
            super().__init__(*args, **kwargs)
            return
        self._db_path = path

        profile_data = _read_xml(path).find('./SecCallPathProfile')
        _LOG.info('%s: %s', path, profile_data.attrib['n'])

        metrics = profile_data.find('./SecHeader/MetricTable')
        _LOG.debug('%s: %s', path, [(_, _.attrib) for _ in metrics])
        self._metrics_by_id = {int(_.attrib['i']): _.attrib['n'] for _ in metrics}
        _LOG.info('%s: %s', path, pprint.pformat(self._metrics_by_id))

        self._metrics_formulas = _derive_metrics_formulas(metrics)
        _LOG.info('%s: %s', path, pprint.pformat(self._metrics_formulas))

        modules = profile_data.find('./SecHeader/LoadModuleTable')
        _LOG.debug('%s: %s', path, [(_, _.attrib) for _ in modules])
        self._modules_by_id = {int(_.attrib['i']): pathlib.Path(_.attrib['n']) for _ in modules}
        _LOG.info('%s: %s', path, pprint.pformat(self._modules_by_id))

        files = profile_data.find('./SecHeader/FileTable')
        _LOG.debug('%s: %s', path, [(_, _.attrib) for _ in files])
        self._files_by_id = {int(_.attrib['i']): pathlib.Path(_.attrib['n']) for _ in files}
        _LOG.info('%s: %s', path, pprint.pformat(self._files_by_id))

        procedures = profile_data.find('./SecHeader/ProcedureTable')
        _LOG.debug('%s: %s', path, [(_, _.attrib) for _ in procedures])
        self._procedures_by_id = {int(_.attrib['i']): _.attrib['n'] for _ in procedures}
        _LOG.info('%s: %s', path, pprint.pformat(self._procedures_by_id))

        measurements = profile_data.find('./SecCallPathProfileData')
        _LOG.debug('%s: %s', path, [(_, _.attrib) for _ in measurements])

        columns = [metric for _, metric in sorted(self._metrics_by_id.items())]

        # customize meanings of columns
        percentage_column = self._determine_percentage_column_base(columns)
        compact_columns = [
            percentage_column + suffix
            for suffix in self._compact_column_suffixes]

        columns += _LOCATION_COLUMNS
        compact_columns += _COMPACT_LOCATION_COLUMNS

        self._meaningful_columns = {
            'percentage': percentage_column,
            'hot_path': percentage_column + self._hot_path_column_suffix,
            'compact': compact_columns}

        rows = self._add_measurements(measurements, max_depth=max_depth)
        index = [_['id'] for _ in rows]
        assert len(index) == len(set(index)), index
        super().__init__(data=rows, index=index, columns=columns)
        self._fix_root_measurement()
        self._add_percentage_columns()

        assert self._meaningful_columns['hot_path'] in self.columns, \
            (self._meaningful_columns['hot_path'], self.columns)
        assert all(_ in self.columns for _ in compact_columns), \
            (compact_columns, self.columns)

    def _evaluate_measurements_data(self, data: dict) -> dict:
        processed_data = {}
        for column, entry in data.items():
            if column not in self._metrics_formulas:
                processed_data[column] = entry
                continue
            formula_code, formula = self._metrics_formulas[column]
            try:
                processed_data[column] = formula(self, data)
            except ValueError as error:
                raise ValueError(
                    '{}: error while evaluating """{}""" to compute "{}" in row {}'
                    .format(self._db_path, formula_code, column, data)) from error
        return processed_data

    def _add_measurements(self, measurements: ET.Element, location: t.Dict[str, t.Any] = None, *,
                          max_depth: t.Optional[int] = None,
                          add_local: bool = True) -> t.List[pd.Series]:
        # split measurements into M and non-M items
        local_measurements = {}
        nonlocal_measurements = []
        for measurement in measurements:
            if measurement.tag != 'M':
                nonlocal_measurements.append(measurement)
            elif add_local:
                local_measurements[self._metrics_by_id[int(measurement.attrib['n'])]] = \
                    float(measurement.attrib['v'])

        if location is None:
            location = {'line': 0, 'id': -1, 'callpath': (), 'type': 'root'}

        if add_local:
            local_measurements = self._evaluate_measurements_data(local_measurements)
            local_measurements.update(location)
            rows = [local_measurements]
        else:
            rows = []

        if max_depth is not None and max_depth <= 0:
            return rows

        for measurement in nonlocal_measurements:
            if measurement.tag not in _MEASUREMENT_TYPES:
                raise NotImplementedError(
                    '{}: measurement type "{}" not recognized:\nattributes={}\nsubnodes={}'
                    .format(self._db_path, measurement.tag, measurement.attrib,
                            [_ for _ in measurement]))

            if self._skip_callsite and measurement.tag == 'C':
                rows += self._add_measurements(measurement, location,
                                               max_depth=max_depth, add_local=False)
                continue

            new_location = {}
            new_location.update(location)
            for (attrib, field), transformer in _LOCATION_DATA_TRANSFORMERS.items():
                if attrib not in measurement.attrib:
                    continue
                if field is None:
                    field = _LOCATION_TYPES[attrib]
                new_location[field] = transformer(self, measurement.attrib[attrib])

            assert 'id' in new_location, \
                (measurement.tag, measurement.attrib, [_ for _ in measurement])
            new_location['type'] = _MEASUREMENT_TYPES[measurement.tag]
            new_location['callpath'] = (*location['callpath'], new_location['id'])

            rows += self._add_measurements(
                measurement, new_location, max_depth=None if max_depth is None else max_depth - 1,
                add_local=True)

        return rows

    def _fix_root_measurement(self):
        pattern = re.compile(r'(?P<prefix>.+:.+) \(E\)')
        column_pairs = []
        for target_column in self.columns:
            match = pattern.fullmatch(target_column)
            if match is None:
                continue
            source_column = '{} (I)'.format(match['prefix'])
            if source_column in self.columns:
                column_pairs.append((target_column, source_column))
                continue
            _LOG.warning('%s: no target column "%s" found for "%s", cannot fix root measurement',
                         self._db_path, target_column, source_column)
        for target_column, source_column in column_pairs:
            self.at[_ROOT_INDEX, target_column] = self.at[_ROOT_INDEX, source_column]

    def _determine_percentage_column_base(self, columns) -> str:
        percentage_column = None
        for candidate in self._percentage_column_candidates:
            col = self._fundamental_column_prefix + candidate
            if col in columns:
                percentage_column = col
                break
        if percentage_column is None:
            unique_prefixes = list(ordered_set.OrderedSet(_.partition(':')[0] for _ in columns))
            _LOG.warning(
                '%s: percentage column candidates %s%s not found, trying %s:%s', self._db_path,
                self._fundamental_column_prefix, self._percentage_column_candidates,
                unique_prefixes, self._percentage_column_candidates)
            for prefix in unique_prefixes:
                for candidate in self._percentage_column_candidates:
                    col = '{}:{}'.format(prefix, candidate)
                    if col in columns:
                        percentage_column = col
                        break
                if percentage_column is not None:
                    break
        assert percentage_column is not None, columns
        return percentage_column

    def _add_percentage_columns(
            self, columns_and_methods: t.Sequence[t.Tuple[str, str]] = None) -> None:
        if columns_and_methods is None:
            base_column = self._meaningful_columns['percentage']
            columns_and_methods = ((base_column, 'total'), (base_column, 'parent'))
        for base_column, method in columns_and_methods:
            self.add_ratio_column(
                base_column, '{} ratio of {}'.format(base_column, method), method)

    def add_ratio_column(self, base_column: str, column_name: str, method: str) -> None:
        """Add a new column with ratio information taken from the base column.

        There are two methods of calculating the ratios.

        "total"
        Compare all values in the base column to the value in that column at the root.

        "parent"
        Compare each value in the base column to the value in that column
        at one level higher in the call path.
        """
        assert base_column in self.columns, (base_column, self.columns)
        assert column_name not in self.columns, (column_name, self.columns)
        column_index = self.columns.get_loc(base_column) + 1
        simple_self = self[[base_column, 'callpath']]
        if method == 'total':
            filtered = simple_self.loc[[_ROOT_INDEX]]
            total = filtered[base_column].item()
            data = [row.at[base_column] / total for _, row in simple_self.iterrows()]
        else:
            assert method == 'parent'
            data = []
            _cache = {}
            for _, row in simple_self.iterrows():
                value = row.at[base_column]
                base_callpath = row.at['callpath']
                base = None
                while base is None or base < value:
                    base_callpath = base_callpath[:-1]
                    if base_callpath in _cache:
                        base = _cache[base_callpath]
                        break
                    try:
                        filtered = simple_self.loc[simple_self['callpath'] == base_callpath]
                    except KeyError:
                        _LOG.exception('%s: no measurements for callpath %s',
                                       self._db_path, base_callpath)
                        continue
                    assert len(filtered) == 1, \
                        (base_column, row.at['callpath'], base_callpath, filtered)
                    base = filtered[base_column].item()
                    _cache[base_callpath] = base
                data.append(value / base)
            del _cache
        self.insert(column_index, column_name, data)

    @property
    def compact(self):
        return self[self._meaningful_columns['compact']]

    def at_paths(self, *fragments, prefix: tuple = (), suffix: tuple = ()) -> pd.DataFrame:
        mask = self.apply(_callpath_filter, axis=1, args=(fragments, prefix, suffix))
        return self[mask]

    def at_depths(self, min_depth: t.Optional[int] = None,
                  max_depth: t.Optional[int] = None) -> pd.DataFrame:
        mask = self.apply(_depth_filter, axis=1, args=(min_depth, max_depth))
        return self[mask]

    def at_depth(self, depth: int) -> pd.DataFrame:
        return self.at_depths(depth, depth)

    def hot_path(self, callpath: t.Sequence[int] = (), threshold: int = 0.05,
                 base_column: str = None) -> pd.DataFrame:
        if base_column is None:
            base_column = self._meaningful_columns['hot_path']
        assert base_column in self.columns, (base_column, self.columns)
        simple_self = self[[base_column, 'callpath']]
        hot_callpaths = []

        while True:
            hot_callpaths.append(callpath)

            simple_self = simple_self.at_paths(prefix=callpath)
            _LOG.debug('%s: %i at target callpath', self._db_path, len(simple_self))
            at_depth = simple_self.at_depth(len(callpath) + 1)
            _LOG.debug('%s: %i at depth %i', self._db_path, len(at_depth), len(callpath) + 1)

            if at_depth.empty:
                break

            hottest_index = at_depth[base_column].idxmax()
            hottest_row = simple_self.loc[hottest_index]
            callpath = hottest_row.at['callpath']
            if hottest_row.at[base_column] < threshold:
                break

        return self[self.callpath.isin(hot_callpaths)]

    def flame_graph(
            self, prefix=(), column='CPUTIME (usec):Mean (I) ratio of parent',
            min_depth=None, max_depth=None,
            shape: str = 'rect', style: str = 'flame', highlight=None):
        """Stack trace graph.

        shape: 'rect' or 'wheel'
        style: 'flame', 'skyline', 'mountains'
        """
        import matplotlib.pyplot as plt
        if min_depth is None:
            min_depth = len(prefix) + 1
        assert min_depth > len(prefix), min_depth
        assert shape in {'rect', 'wheel'}, shape
        if style == 'flame':
            color_map = plt.get_cmap('autumn')
            colors = lambda n: color_map(np.linspace(0, 1, n))
        elif style == 'skyline':
            color_map = plt.get_cmap('YlGnBu')
            colors = lambda n: color_map(np.linspace(0, 1, n))
        elif style == 'mountains':
            color_map = plt.get_cmap('Greys')
            colors = lambda n: color_map(np.linspace(0, 1, n))
        else:
            color_map = plt.get_cmap('tab20c')
            colors = lambda n: color_map(np.arange(n))
        fig, ax = plt.subplots(subplot_kw=dict(polar=shape == 'wheel'), figsize=(16, 16))
        thickness = 1

        at_depth = {}

        base = self.at_paths(prefix=prefix)

        # for depth in range(min_depth, max_depth + 1):
        depth = min_depth
        while max_depth is None or depth <= max_depth:
            _LOG.info('at depth %i', depth)
            at_depth[depth] = {}
            df = base.at_depth(depth)
            at_depth[depth]['df'] = df
            if df.empty:
                break

            ids = at_depth[depth]['df']['id'].values
            _LOG.info('ids:  %s', ids)
            raw_values = at_depth[depth]['df'][column].values
            _LOG.debug('raw:  %s', raw_values)

            if depth - 1 in at_depth:
                # normalize data to previous layer
                by_parent = {}
                for i, (_, series) in enumerate(at_depth[depth]['df'].iterrows()):
                    callpath = series['callpath']
                    parent = callpath[-2]
                    if parent not in by_parent:
                        by_parent[parent] = []
                    by_parent[parent].append((i, series))
                _LOG.debug('by parent: %s', {_: [i for i, series in items]
                                             for _, items in by_parent.items()})

                normalized_values = []
                offsets = []
                for parent, items in by_parent.items():
                    ratio = at_depth[depth - 1]['widths'][parent] / _NORMALIZATION_CONSTANT
                    raw_items = np.array([raw_values[i] for i, _ in items])
                    normalized_items = raw_items / np.sum(raw_items) * _NORMALIZATION_CONSTANT * ratio
                    normalized_values += list(normalized_items)

                    base_offest = at_depth[depth - 1]['offsets'][parent]
                    items_offsets = np.append(0, normalized_items.cumsum()[:-1]) + base_offest
                    assert len(normalized_items) == len(items_offsets)
                    offsets += list(items_offsets)
                widths = np.array(normalized_values)
                offsets = np.array(offsets)
                assert len(widths) == len(offsets)
            else:
                widths = raw_values / np.sum(raw_values) * _NORMALIZATION_CONSTANT
                _LOG.debug('norm const: %f', _NORMALIZATION_CONSTANT)
                offsets = np.append(0, widths.cumsum()[:-1])
            at_depth[depth]['offsets'] = collections.OrderedDict(
                [(id_, _) for id_, _ in zip(ids, offsets)])
            _LOG.info('offsets: %s', offsets)

            at_depth[depth]['widths'] = collections.OrderedDict(
                [(id_, _) for id_, _ in zip(ids, widths)])
            _LOG.info('widths: %s', widths)

            y = (depth - min_depth + 1) * thickness
            ax.bar(
                x=offsets, width=widths, bottom=y, height=thickness,
                color=colors(len(offsets)), edgecolor='w', linewidth=1, align='edge')

            for i, id_ in enumerate(ids):
                if widths[i] < np.pi / (depth - min_depth + 32):
                    continue
                x = offsets[i] + widths[i] / 2
                if shape == 'wheel':
                    rotation = x * 180 / np.pi - 90
                else:
                    rotation = 0
                # text = str(id_)
                text = self.loc[id_]['procedure']
                ax.text(
                    x=x, y=y + thickness * 0.2, s=text,
                    rotation=rotation,
                    horizontalalignment='center', verticalalignment='center')
            depth += 1

        ax.set(title=self._db_path.name)
        ax.set_axis_off()
        plt.show()
