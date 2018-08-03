#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

from collections import namedtuple
from configparser import ConfigParser
from enum import Enum
from glob import glob
from typing import Any, List, Union
import logging as log
import os.path
import sys
from astropy.io import fits
from numba import jit
import click
import numpy as np

__version__ = '1.1.1'


class FlagType(Enum):
    equality = 0
    bitwise_and = 1


class FlagAction(Enum):
    include = 0
    exclude = 1


Flagging = namedtuple('Flagging', 'flag_type flag_value flag_action')


def flag_mask(flags: Any,
              flagging: Flagging) -> Any:
    if flagging.flag_type == FlagType.equality:
        mask = (flags == flagging.flag_value)
    elif flagging.flag_type == FlagType.bitwise_and:
        mask = (np.array(np.bitwise_and(flags, flagging.flag_value),
                         dtype='bool') != 0)
    else:
        raise ValueError('unknown FlagType in "flag_mask": {0}'
                         .format(flagging.flag_type))

    if flagging.flag_action == FlagAction.exclude:
        mask = np.logical_not(mask)

    return mask


IndexConfiguration = namedtuple('IndexConfiguration',
                                ['flagging',
                                 'flag_hdu',
                                 'flag_column',
                                 'input_path',
                                 'input_mask',
                                 'input_hdu',
                                 'input_column',
                                 'period_length',
                                 'output_file_name'])
TODFileInfo = namedtuple('TODFileInfo',
                         ['file_name',
                          'mod_time',
                          'num_of_samples',
                          'num_of_unflagged_samples'])


class IndexFile:
    def __init__(self,
                 input_hdu: Union[int, str]=1,
                 input_column: Union[int, str]=0,
                 flag_hdu: Union[int, str]=1,
                 flag_column: Union[int, str]=1,
                 flagging: Flagging=None):
        self.flagging = flagging
        self.tod_info = []
        self.periods = None
        self.period_length = 0.0

        self.input_hdu = input_hdu
        self.input_column = input_column
        self.flag_hdu = flag_hdu
        self.flag_column = flag_column

    def store_in_hdus(self) -> List:
        fname_format = '{0}A'.format(
            max([len(x.file_name) for x in self.tod_info]))
        file_info_columns = [fits.Column(name='FILENAME',
                                         format=fname_format,
                                         array=[x.file_name for x in self.tod_info]),
                             fits.Column(name='MODTIME',
                                         format='1D',
                                         array=[x.mod_time for x in self.tod_info]),
                             fits.Column(name='NSAMPLES',
                                         format='1K',
                                         array=[x.num_of_samples
                                                for x in self.tod_info]),
                             fits.Column(name='NUNFLAG',
                                         format='1K',
                                         array=[x.num_of_unflagged_samples
                                                for x in self.tod_info])]
        file_info_hdu = fits.BinTableHDU.from_columns(file_info_columns)
        file_info_hdu.name = 'FILEINFO'
        period_columns = [fits.Column(name='NSAMPLES',
                                      format='1K',
                                      array=self.periods)]
        period_hdu = fits.BinTableHDU.from_columns(period_columns)
        period_hdu.name = 'PERIODS'
        if self.flagging is not None:
            list_of_flag_types = ', '.join(['"{0}"'.format(x.name)
                                            for x in FlagType])
            list_of_flag_actions = ', '.join(['"{0}"'.format(x.name)
                                              for x in FlagAction])

            file_info_hdu.header['FTYPE'] = (self.flagging.flag_type.name,
                                             list_of_flag_types)
            file_info_hdu.header['FVALUE'] = (self.flagging.flag_value,
                                              'Value used for flagging')
            file_info_hdu.header['FACTION'] = (self.flagging.flag_action.name,
                                               'Flag action ({0})'
                                               .format(list_of_flag_actions))
            file_info_hdu.header['FHDU'] = (self.flag_hdu,
                                            'Flags column HDU')
            file_info_hdu.header['FCOL'] = (self.flag_column,
                                            'Flags column number/name')

        file_info_hdu.header['INPHDU'] = (self.input_hdu,
                                          'Time column HDU')
        file_info_hdu.header['INPCOL'] = (self.input_column,
                                          'Time column number/name')
        file_info_hdu.header['NSAMPLES'] = (sum([x.num_of_samples
                                                 for x in self.tod_info]),
                                            'Total number of samples')
        file_info_hdu.header['NUNFLAG'] = (sum([x.num_of_unflagged_samples
                                                for x in self.tod_info]),
                                           'Total number of unflagged samples')

        period_hdu.header['LENGTH'] = (
            self.period_length, 'Length of a period')
        return [file_info_hdu, period_hdu]

    def load_from_fits(self, file_name: str, first_idx=-1, last_idx=-1):
        with fits.open(file_name) as f:
            fileinfo_hdu = f['FILEINFO']
            fileinfo_hdr = fileinfo_hdu.header
            if 'FTYPE' in fileinfo_hdr:
                self.flagging = \
                    Flagging(flag_type=FlagType[fileinfo_hdr['FTYPE']],
                             flag_action=FlagAction[fileinfo_hdr['FACTION']],
                             flag_value=fileinfo_hdr['FVALUE'])
                self.flag_hdu = int_or_str(fileinfo_hdr['FHDU'])
                self.flag_column = int_or_str(fileinfo_hdr['FCOL'])
            else:
                self.flagging = self.flag_hdu = self.flag_column = None

            self.tod_info = [TODFileInfo(file_name=x[0].decode(),
                                         mod_time=x[1],
                                         num_of_samples=x[2],
                                         num_of_unflagged_samples=x[3])
                             for x in fileinfo_hdu.data.tolist()]
            if first_idx < 0:
                _first = 0
            else:
                _first = first_idx

            if last_idx < 0:
                _last = len(self.tod_info) - 1
            else:
                _last = min(len(self.tod_info) - 1, last_idx)

            self.tod_info = self.tod_info[_first:(_last + 1)]
            self.input_hdu = fileinfo_hdu.header['INPHDU']
            self.input_column = fileinfo_hdu.header['INPCOL']

            periods_hdu = f['PERIODS']
            self.periods = periods_hdu.data.field('NSAMPLES')
            self.period_length = periods_hdu.header['LENGTH']


@jit
def split_into_periods(time_array, period_length, periods):
    '''Decide the length (in samples) of the destriping periods,
    according to the timing of each sample. Both "time_array" and
    "period length" must be expressed using the same measure unit
    (e.g., seconds, clock ticks...). The array "periods" must have
    been sized before calling this function.'''

    periods[:] = 0
    sample_idx = 0
    period_idx = 0
    while sample_idx < len(time_array):
        start_time = time_array[sample_idx]
        while (sample_idx < len(time_array)) and \
              (time_array[sample_idx] - start_time < period_length):
            sample_idx += 1
            periods[period_idx] += 1

        period_idx += 1


def read_index_conf_file(file_name: str) -> IndexConfiguration:
    conf_file = ConfigParser()
    conf_file.read(file_name)

    try:
        try:
            flag_section = conf_file['flagging']
        except KeyError:
            flag_section = flagging = flag_hdu = flag_column = None

        if flag_section is not None:
            flagging = Flagging(flag_type=FlagType[flag_section.get('type')],
                                flag_value=flag_section.getint('value'),
                                flag_action=FlagAction[flag_section.get('action')])
            flag_hdu = int_or_str(flag_section.get('hdu'))
            flag_column = int_or_str(flag_section.get('column'))
        try:
            input_section = conf_file['input_files']
            output_section = conf_file['output_file']
            period_section = conf_file['periods']

            input_path = input_section.get('path', fallback='.')
            time_hdu = int_or_str(input_section.get('hdu', fallback=1))
            time_column = int_or_str(input_section.get('column', fallback=1))
            period_length = period_section.getfloat('length')
            output_file_name = output_section.get('file_name')
        except KeyError as e:
            log.error('section/key %s not found in the configuration file "%s"',
                      e, file_name)
            sys.exit(1)
    except ValueError as e:
        log.error('invalid value found in one of the entries in "%s": %s',
                  file_name, e)
        sys.exit(1)

    return IndexConfiguration(flagging=flagging,
                              flag_hdu=flag_hdu,
                              flag_column=flag_column,
                              input_path=input_path,
                              input_mask=input_section.get('mask'),
                              input_hdu=time_hdu,
                              input_column=time_column,
                              period_length=period_length,
                              output_file_name=output_file_name)


def int_or_str(x: str) -> Union[int, str]:
    'Convert "x" into an integer if possible. Otherwise, return it unmodified.'
    try:
        int_value = int(x)
        return int_value
    except ValueError:
        return x  # A string


def write_output(file_name: str,
                 info_list: List[TODFileInfo],
                 periods: Any,
                 configuration: IndexConfiguration):
    log.info('writing file "%s"', file_name)

    index_file = IndexFile(input_hdu=configuration.input_hdu,
                           input_column=configuration.input_column,
                           flag_hdu=configuration.flag_hdu,
                           flag_column=configuration.flag_column,
                           flagging=configuration.flagging)
    index_file.tod_info = info_list
    index_file.periods = periods
    index_file.period_length = configuration.period_length

    hdu_list = [fits.PrimaryHDU()] + index_file.store_in_hdus()
    fits.HDUList(hdu_list).writeto(file_name, overwrite=True)
    log.info('file "%s" written successfully', file_name)


def read_column(file_name: str,
                hdu: Union[int, str],
                column: Union[int, str]):
    with fits.open(file_name) as f:
        return f[hdu].data.field(column)


@click.command()
@click.argument('configuration_file')
def index_main(configuration_file):
    log.basicConfig(level=log.INFO,
                    format='[%(asctime)s %(levelname)s] %(message)s')
    log.info('reading configuration file "%s"', configuration_file)
    configuration = read_index_conf_file(configuration_file)
    log.info('configuration file read successfully')
    list_of_file_names = \
        sorted(glob(os.path.join(os.path.expanduser(configuration.input_path),
                                 configuration.input_mask)))

    list_of_file_info = []  # type: List[TODFileInfo]
    periods = np.array([], dtype='int64')
    prev_times = None
    prev_flags = None
    prev_last_time = None
    for idx, file_name in enumerate(list_of_file_names):
        log.info('processing file "%s" (%d/%d)',
                 file_name, idx + 1, len(list_of_file_names))

        times = read_column(file_name=file_name,
                            hdu=configuration.input_hdu,
                            column=configuration.input_column)

        if (prev_last_time is not None) and (times[0] < prev_last_time):
            log.error(('column {0} in HDU {1} of the FITS files is not '
                       'sorted in ascending order ({2} > {3})')
                      .format(configuration.input_column,
                              configuration.input_hdu,
                              times[0], prev_last_time))
            sys.exit(1)

        prev_last_time = times[-1]
        num_of_samples = len(times)

        if configuration.flagging:
            flags = read_column(file_name=file_name,
                                hdu=configuration.flag_hdu,
                                column=configuration.flag_column)

            mask = flag_mask(flags=flags,
                             flagging=configuration.flagging)
            times = times[mask]

        file_info = TODFileInfo(file_name=os.path.abspath(file_name),
                                mod_time=os.path.getmtime(file_name),
                                num_of_samples=num_of_samples,
                                num_of_unflagged_samples=len(times))
        list_of_file_info.append(file_info)
        if prev_times is not None:
            times = np.concatenate((prev_times, times))
            if configuration.flagging is not None:
                flags = np.concatenate((prev_flags, flags))

        num_of_periods = int((times[-1] - times[0]) //
                             configuration.period_length) + 1
        cur_periods = np.zeros(num_of_periods, dtype='int64')
        split_into_periods(times, configuration.period_length, cur_periods)
        cur_periods = cur_periods[cur_periods > 0]

        if idx + 1 < len(list_of_file_names):
            last_period_len = cur_periods[-1]
            prev_times = times[-last_period_len:]
            if configuration.flagging is not None:
                prev_flags = flags[-last_period_len:]

            cur_periods = cur_periods[0:-1]
        periods = np.concatenate((periods, cur_periods))

        log.info('file "%s" processed successfully', file_name)

    write_output(file_name=configuration.output_file_name,
                 info_list=list_of_file_info,
                 periods=periods,
                 configuration=configuration)


if __name__ == '__main__':
    index_main()
