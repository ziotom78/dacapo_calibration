#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import io
import sys
from copy import copy
from collections import namedtuple
from configparser import ConfigParser
from datetime import datetime

import click
from typing import List, Any
import numpy as np
import scipy
import healpy
import logging as log
from numba import jit
from index import int_or_str, flag_mask, TODFileInfo, IndexFile
from astropy.io import fits
from mpi4py import MPI
import ftnroutines

__version__ = '1.1.1'


class Profiler:
    def __init__(self):
        self.start_time = None
        self.tic()

    def tic(self):
        '''Record the current time.'''
        self.start_time = datetime.now()

    def toc(self):
        '''Return the elapsed time (in seconds) since the last call to tic/toc.'''
        now = datetime.now()
        diff = now - self.start_time
        self.start_time = now
        return diff.total_seconds()


def gather_arrays(mpi_comm, array: Any, root=0) -> Any:
    lengths = mpi_comm.gather(len(array), root=root)
    if mpi_comm.Get_rank() == root:
        recvbuf = np.empty(sum(lengths), dtype=array.dtype)
    else:
        recvbuf = None

    mpi_comm.Gatherv(sendbuf=array, recvbuf=(recvbuf, lengths), root=root)
    return recvbuf


CalibrateConfiguration = namedtuple('CalibrateConfiguration',
                                    ['index_file',
                                     'first_tod_index', 'last_tod_index',
                                     'signal_hdu', 'signal_column',
                                     'pointing_hdu', 'pointing_columns',
                                     't_cmb_k', 'solsys_speed_vec_m_s',
                                     'frequency_hz',
                                     'nside', 'mask_file_path',
                                     'periods_per_cal_constant',
                                     'cg_stop', 'cg_maxiter',
                                     'dacapo_stop', 'dacapo_maxiter',
                                     'pcond', 'output_file_name', 'save_map',
                                     'save_convergence',
                                     'comment',
                                     'parameter_file_contents'])


class OfsAndGains:
    def __init__(self, offsets, gains, samples_per_ofsp, samples_per_gainp):
        self.a_vec = np.concatenate((offsets, gains))
        self.samples_per_ofsp = np.array(samples_per_ofsp, dtype='int')
        self.samples_per_gainp = np.array(samples_per_gainp, dtype='int')

        self.ofsp_per_gainp = OfsAndGains.calc_ofsp_per_gainp(samples_per_ofsp,
                                                              samples_per_gainp)

    def __copy__(self):
        return OfsAndGains(offsets=np.copy(self.offsets),
                           gains=np.copy(self.gains),
                           samples_per_ofsp=np.copy(self.samples_per_ofsp),
                           samples_per_gainp=np.copy(self.samples_per_gainp))

    def __repr__(self):
        return 'a: {0} (offsets: {1}, gains: {2})'.format(self.a_vec,
                                                          self.offsets,
                                                          self.gains)

    @property
    def offsets(self):
        return self.a_vec[0:len(self.samples_per_ofsp)]

    @property
    def gains(self):
        return self.a_vec[len(self.samples_per_ofsp):]

    @staticmethod
    def calc_ofsp_per_gainp(samples_per_ofsp, samples_per_gainp):
        log.debug('entering calc_ofsp_per_gainp')

        ofsp_per_gainp = []

        cur_ofsp_idx = 0
        for samples_in_cur_gainp in samples_per_gainp:
            ofsp_in_cur_gainp = 0
            sample_count = 0
            while sample_count < samples_in_cur_gainp:
                sample_count += samples_per_ofsp[cur_ofsp_idx]
                cur_ofsp_idx += 1
                ofsp_in_cur_gainp += 1

            assert sample_count == samples_in_cur_gainp
            ofsp_per_gainp.append(ofsp_in_cur_gainp)

        return ofsp_per_gainp


def ofs_and_gains_with_same_lengths(source: OfsAndGains, a_vec):
    result = copy(source)
    result.a_vec = np.copy(a_vec)
    return result


class MonopoleAndDipole:
    def __init__(self, mask, dipole_map):
        if mask is not None:
            self.monopole_map = np.array(mask, dtype='float')
        else:
            self.monopole_map = np.ones_like(dipole_map)

        self.dipole_map = np.array(dipole_map) * np.array(self.monopole_map)


def split_into_n(length: int, num_of_segments: int):
    log.debug('entering split_into_n')
    assert (num_of_segments > 0), \
        "num_of_segments={0} is not positive".format(num_of_segments)
    assert (length >= num_of_segments), \
        "length={0} is smaller than num_of_segments={1}".format(
            length, num_of_segments)

    start_positions = np.array([int(i * length / num_of_segments)
                                for i in range(num_of_segments + 1)],
                               dtype='int')
    return start_positions[1:] - start_positions[0:-1]


def split(length, sublength: int):
    log.debug('entering split')
    assert (sublength > 0), "sublength={0} is not positive".format(sublength)
    assert (sublength < length), \
        "sublength={0} is not smaller than length={1}".format(
            sublength, length)

    return split_into_n(length=length,
                        num_of_segments=int(np.ceil(length / sublength)))


class TOD:
    def __init__(self, signal, pix_idx, num_of_pixels):
        self.signal = signal
        self.pix_idx = pix_idx
        self.num_of_pixels = num_of_pixels


TODSubrange = namedtuple('TODSubrange',
                         ['file_info',
                          'first_idx',
                          'num_of_samples'])


def assign_files_to_processes(samples_per_process: Any,
                              tod_info: List[TODFileInfo]) -> List[List[TODSubrange]]:
    log.debug('entering assign_files_to_processes')

    result = []  # type: List[List[TODSubrange]]
    file_idx = 0
    file_sample_idx = 0
    samples_in_file = tod_info[file_idx].num_of_unflagged_samples
    for samples_for_this_MPI_proc in samples_per_process:
        samples_left = samples_for_this_MPI_proc
        MPI_proc_subranges = []
        while (samples_left > 0) and (file_idx < len(tod_info)):
            if samples_in_file > samples_left:
                MPI_proc_subranges.append(TODSubrange(file_info=tod_info[file_idx],
                                                      first_idx=file_sample_idx,
                                                      num_of_samples=samples_left))
                file_sample_idx += samples_left
                samples_in_file -= samples_left
                samples_left = 0
            else:
                MPI_proc_subranges.append(TODSubrange(file_info=tod_info[file_idx],
                                                      first_idx=file_sample_idx,
                                                      num_of_samples=samples_in_file))
                samples_left -= samples_in_file
                samples_in_file = 0

            if samples_in_file == 0:
                if file_idx + 1 == len(tod_info):
                    break  # No more files, exit the while loop

                file_sample_idx = 0
                file_idx += 1
                samples_in_file = tod_info[file_idx].num_of_unflagged_samples

        result.append(MPI_proc_subranges)

    return result


def load_subrange(subrange: TODSubrange,
                  index: IndexFile,
                  configuration: CalibrateConfiguration) -> TOD:
    log.debug('entering load_subrange')

    with fits.open(subrange.file_info.file_name) as f:
        signal = f[configuration.signal_hdu].data.field(
            configuration.signal_column)
        if len(signal) != subrange.file_info.num_of_samples:
            log.error('expected %d samples in file "%s", but %d found: '
                      'you should rebuild the index file',
                      subrange.file_info.num_of_samples,
                      subrange.file_info.file_name,
                      len(signal))
            sys.exit(1)

        pix_idx = [f[configuration.pointing_hdu].data.field(x)
                   for x in configuration.pointing_columns]
        if len(pix_idx) == 2:
            theta, phi = pix_idx
            pix_idx = healpy.ang2pix(configuration.nside, theta, phi)
            del theta, phi
        elif len(configuration.pointing_columns) == 1:
            pix_idx = pix_idx[0]
        else:
            log.error('one or two columns are expected for the pointings (got %s)',
                      ', '.join([str(x) for x in configuration.pointing_columns]))
            sys.exit(1)
        if index.flagging is not None:
            flags = f[index.flag_hdu].data.field(index.flag_column)
            mask = flag_mask(flags, index.flagging)

            signal = signal[mask]
            pix_idx = pix_idx[mask]

    if len(signal) != subrange.file_info.num_of_unflagged_samples:
        log.error('expected %d unflagged samples in file "%s", but %d found: '
                  'you should rebuild the index file',
                  subrange.file_info.num_of_unflagged_samples,
                  subrange.file_info.file_name,
                  len(signal))
        sys.exit(1)

    start, end = (subrange.first_idx,
                  (subrange.first_idx + subrange.num_of_samples))
    signal = signal[start:end]
    pix_idx = pix_idx[start:end]

    return TOD(signal=signal, pix_idx=pix_idx,
               num_of_pixels=healpy.nside2npix(configuration.nside))


def load_tod(tod_list: List[TODSubrange],
             index: IndexFile,
             configuration: CalibrateConfiguration) -> TOD:
    log.debug('entering load_tod')

    result = TOD(signal=np.array([], dtype='float'),
                 pix_idx=np.array([], dtype='int'),
                 num_of_pixels=healpy.nside2npix(configuration.nside))
    for cur_subrange in tod_list:
        cur_tod = load_subrange(cur_subrange, index, configuration)
        result.signal = np.concatenate((result.signal, cur_tod.signal))
        result.pix_idx = np.concatenate((result.pix_idx, cur_tod.pix_idx))

    return result


def read_calibrate_conf_file(file_name: str) -> CalibrateConfiguration:
    log.debug('entering read_calibrate_conf_file')
    conf_file = ConfigParser()
    conf_file.read(file_name)

    try:
        input_sect = conf_file['input_files']

        index_file = input_sect.get('index_file', None)
        first_tod_index = input_sect.get('first_tod_index', -1)
        last_tod_index = input_sect.get('last_tod_index', -1)
        signal_hdu = int_or_str(input_sect.get('signal_hdu'))
        signal_column = int_or_str(input_sect.get('signal_column'))
        pointing_hdu = int_or_str(input_sect.get('pointing_hdu'))
        pointing_columns = [int_or_str(x.strip())
                            for x in input_sect.get('pointing_columns').split(',')]
        dacapo_sect = conf_file['dacapo']

        t_cmb_k = dacapo_sect.getfloat('t_cmb_k')
        solsysdir_ecl_colat_rad = dacapo_sect.getfloat(
            'solsysdir_ecl_colat_rad')
        solsysdir_ecl_long_rad = dacapo_sect.getfloat('solsysdir_ecl_long_rad')
        solsysspeed_m_s = dacapo_sect.getfloat('solsysspeed_m_s')
        solsys_speed_vec_m_s = solsysspeed_m_s * \
            np.array([np.sin(solsysdir_ecl_colat_rad) * np.cos(solsysdir_ecl_long_rad),
                      np.sin(solsysdir_ecl_colat_rad) *
                      np.sin(solsysdir_ecl_long_rad),
                      np.cos(solsysdir_ecl_colat_rad)])

        freq_str = dacapo_sect.get('frequency_hz', fallback='none')
        if freq_str.lower() in ['', 'none', 'nan', 'no']:
            frequency_hz = None
        else:
            frequency_hz = float(freq_str)

        nside = dacapo_sect.getint('nside')
        mask_file_path = dacapo_sect.get('mask', fallback=None)
        if mask_file_path.strip() == '':
            mask_file_path = None

        periods_per_cal_constant = dacapo_sect.getint(
            'periods_per_cal_constant')
        cg_stop = dacapo_sect.getfloat('cg_stop_value', 1e-9)
        cg_maxiter = dacapo_sect.getint('cg_max_iterations', 100)
        dacapo_stop = dacapo_sect.getfloat('dacapo_stop_value', 1e-9)
        dacapo_maxiter = dacapo_sect.getint('dacapo_max_iterations', 20)
        pcond = dacapo_sect.get('pcond').lower()
        try:
            if not healpy.isnsideok(nside):
                raise ValueError('invalid NSIDE = {0}'.format(nside))
            if cg_stop < 0.0:
                raise ValueError('cg_stop_value ({0:.3e}) should not be negative'
                                 .format(cg_stop))
            if dacapo_stop < 0:
                raise ValueError('dacapo_stop_value ({0:.3e}) should not be negative'
                                 .format(dacapo_stop))
            if periods_per_cal_constant < 1:
                raise ValueError('periods_per_cal_constant ({0}) should be greater than zero'
                                 .format(periods_per_cal_constant))
            if cg_maxiter < 0:
                raise ValueError(
                    'cg_maxiter (%d) cannot be negative'.format(cg_maxiter))
            if dacapo_maxiter < 0:
                raise ValueError(
                    'dacapo_maxiter (%d) cannot be negative'.format(dacapo_maxiter))
        except ValueError as e:
            log.error(e)
            sys.exit(1)
        output_sect = conf_file['output']

        output_file_name = output_sect.get('file_name')
        save_map = output_sect.getboolean('save_map', fallback=True)
        save_convergence = output_sect.getboolean(
            'save_convergence_information', fallback=True)
        comment = output_sect.get('comment', fallback=None)
    except ValueError as e:
        log.error('invalid value found in one of the entries in "%s": %s',
                  file_name, e)

    param_file_contents = io.StringIO()
    conf_file.write(param_file_contents)
    param_file_contents = np.array(
        list(param_file_contents.getvalue().encode('utf-8')))
    return CalibrateConfiguration(index_file=index_file,
                                  first_tod_index=first_tod_index,
                                  last_tod_index=last_tod_index,
                                  signal_hdu=signal_hdu,
                                  signal_column=signal_column,
                                  pointing_hdu=pointing_hdu,
                                  pointing_columns=pointing_columns,
                                  t_cmb_k=t_cmb_k,
                                  solsys_speed_vec_m_s=solsys_speed_vec_m_s,
                                  frequency_hz=frequency_hz,
                                  nside=nside,
                                  mask_file_path=mask_file_path,
                                  periods_per_cal_constant=periods_per_cal_constant,
                                  cg_stop=cg_stop,
                                  cg_maxiter=cg_maxiter,
                                  dacapo_stop=dacapo_stop,
                                  dacapo_maxiter=dacapo_maxiter,
                                  pcond=pcond,
                                  output_file_name=output_file_name,
                                  save_map=save_map,
                                  save_convergence=save_convergence,
                                  comment=comment,
                                  parameter_file_contents=param_file_contents)


SPEED_OF_LIGHT_M_S = 2.99792458e8
PLANCK_H_MKS = 6.62606896e-34
BOLTZMANN_K_MKS = 1.3806504e-23


def get_dipole_temperature(t_cmb_k: float, solsys_speed_vec_m_s, directions, freq=None):
    '''Given one or more one-length versors, return the intensity of the CMB dipole

    The vectors must be expressed in the Ecliptic coordinate system.
    If "freq" (frequency in Hz) is specified, the formulation will use the
    quadrupolar correction.
    '''
    log.debug('entering get_dipole_temperature')

    beta = solsys_speed_vec_m_s / SPEED_OF_LIGHT_M_S
    if freq:
        fact = PLANCK_H_MKS * freq / (BOLTZMANN_K_MKS * t_cmb_k)
        expfact = np.exp(fact)
        q = (fact / 2) * (expfact + 1) / (expfact - 1)
        dotprod = np.dot(beta, directions)
        return t_cmb_k * (dotprod + q * dotprod**2)
    else:
        gamma = (1 - np.dot(beta, beta))**(-0.5)

        return t_cmb_k * (1.0 / (gamma * (1 - np.dot(beta, directions))) - 1.0)


def apply_f(a: OfsAndGains, pix_idx, dipole_map, sky_map):
    log.debug('entering apply_f')
    return ftnroutines.apply_f(a.offsets, a.gains,
                               a.samples_per_ofsp, a.samples_per_gainp,
                               pix_idx, dipole_map, sky_map)


def apply_ft(vector, a: OfsAndGains, pix_idx, dipole_map, sky_map):
    log.debug('entering apply_ft')
    return ftnroutines.apply_ft(vector, a.offsets, a.gains,
                                a.samples_per_ofsp, a.samples_per_gainp,
                                pix_idx, dipole_map, sky_map)


def sum_local_results(mpi_comm, function, **arguments):
    log.debug('entering sum_local_results')
    result = function(**arguments)

    if mpi_comm:
        totals = np.zeros_like(result)
        mpi_comm.Allreduce(sendbuf=result, recvbuf=totals, op=MPI.SUM)
        return totals
    else:
        return result


def compute_diagm_locally(a: OfsAndGains, pix_idx, num_of_pixels: int):
    log.debug('entering compute_diagm_locally')
    result = np.empty(num_of_pixels, dtype='float')
    ftnroutines.compute_diagm_locally(
        a.gains, a.samples_per_gainp, pix_idx, result)
    return result


def compute_diagm(mpi_comm, a: OfsAndGains, pix_idx, num_of_pixels: int):
    log.debug('entering compute_diagm')
    return sum_local_results(mpi_comm, function=compute_diagm_locally,
                             a=a,
                             pix_idx=pix_idx,
                             num_of_pixels=num_of_pixels)


def apply_ptilde(map_pixels, a: OfsAndGains, pix_idx):
    log.debug('entering apply_ptilde')
    return ftnroutines.apply_ptilde(map_pixels, a.gains, a.samples_per_gainp, pix_idx)


def apply_ptildet_locally(vector, a: OfsAndGains, pix_idx, num_of_pixels: int):
    log.debug('entering apply_ptildet_locally')
    result = np.empty(num_of_pixels, dtype='float')
    ftnroutines.apply_ptildet_locally(vector, a.gains,
                                      a.samples_per_gainp,
                                      pix_idx, result)
    return result


def apply_ptildet(mpi_comm, vector, a: OfsAndGains, pix_idx,
                  num_of_pixels: int):
    log.debug('entering apply_ptildet')
    return sum_local_results(mpi_comm, function=apply_ptildet_locally,
                             vector=vector,
                             a=a,
                             pix_idx=pix_idx,
                             num_of_pixels=num_of_pixels)


def apply_z(mpi_comm, vector, a: OfsAndGains, pix_idx, mc: MonopoleAndDipole):
    log.debug('entering apply_z')
    binned_map = apply_ptildet(mpi_comm, vector, a, pix_idx,
                               len(mc.dipole_map))
    diagM = compute_diagm(mpi_comm, a, pix_idx, len(mc.dipole_map))

    nonzero_hit_mask = diagM != 0
    inv_diagM = np.zeros_like(diagM)
    inv_diagM[nonzero_hit_mask] = 1.0 / diagM[nonzero_hit_mask]

    binned_map = np.multiply(binned_map, inv_diagM)
    monopole_dot = np.dot(mc.monopole_map, binned_map)
    dipole_dot = np.dot(mc.dipole_map, binned_map)

    # Compute (m_c^T M^-1 m_c)
    small_matr = np.array([[np.dot(mc.dipole_map,
                                   np.multiply(inv_diagM, mc.dipole_map)),
                            np.dot(mc.dipole_map, inv_diagM)],
                           [np.dot(mc.monopole_map,
                                   np.multiply(inv_diagM, mc.dipole_map)),
                            np.dot(mc.monopole_map, inv_diagM)]])
    small_matr_prod = np.linalg.inv(small_matr) @ np.array([dipole_dot,
                                                            monopole_dot])

    ftnroutines.clean_binned_map(inv_diagM, mc.dipole_map, mc.monopole_map,
                                 small_matr_prod, binned_map)

    return vector - apply_ptilde(binned_map, a, pix_idx)


def apply_A(mpi_comm, a: OfsAndGains, sky_map, pix_idx,
            mc: MonopoleAndDipole, x: OfsAndGains):
    log.debug('entering apply_A')

    vector1 = apply_f(x, pix_idx, mc.dipole_map, sky_map)
    vector2 = apply_z(mpi_comm, vector1, a, pix_idx, mc)
    return apply_ft(vector2, a, pix_idx, mc.dipole_map, sky_map)


def compute_v(mpi_comm, voltages, a: OfsAndGains, sky_map, pix_idx,
              mc: MonopoleAndDipole):
    log.debug('entering compute_v')
    vector = apply_z(mpi_comm, voltages, a, pix_idx, mc)
    return apply_ft(vector, a, pix_idx, mc.dipole_map, sky_map)


def mpi_dot_prod(mpi_comm, x, y):
    log.debug('entering mpi_dot_prod')

    local_sum = np.dot(x, y)

    if mpi_comm is not None:
        return mpi_comm.allreduce(local_sum, op=MPI.SUM)
    else:
        return local_sum


def conjugate_gradient(mpi_comm, voltages, start_a: OfsAndGains, sky_map,
                       pix_idx, mc: MonopoleAndDipole, pcond=None,
                       threshold=1e-9, max_iter=100):

    log.debug('entering conjugate_gradient')

    a = copy(start_a)
    residual = (compute_v(mpi_comm, voltages, a, sky_map, pix_idx, mc) -
                apply_A(mpi_comm, a, sky_map, pix_idx, mc, a))
    r = ofs_and_gains_with_same_lengths(source=start_a,
                                        a_vec=residual)

    k = 0
    list_of_stopping_factors = []
    stopping_factor = np.sqrt(mpi_dot_prod(mpi_comm, r.a_vec, r.a_vec))
    log.info('conjugate_gradient: iteration %d/%d, stopping criterion: %.5e',
             k, max_iter, stopping_factor)

    list_of_stopping_factors.append(stopping_factor)
    if stopping_factor < threshold:
        return a, list_of_stopping_factors
    if pcond is not None:
        z = pcond.apply_to(r)
    else:
        z = copy(r)

    old_r_dot = mpi_dot_prod(mpi_comm, z.a_vec, r.a_vec)
    p = copy(z)
    best_stopping_factor = stopping_factor
    best_a = a

    while True:
        k += 1
        if k >= max_iter:
            return best_a, list_of_stopping_factors

        Ap = apply_A(mpi_comm, a, sky_map, pix_idx, mc, p)
        gamma = old_r_dot / mpi_dot_prod(mpi_comm, p.a_vec, Ap)
        a.a_vec += gamma * p.a_vec
        r.a_vec -= gamma * Ap

        stopping_factor = np.sqrt(mpi_dot_prod(mpi_comm, r.a_vec, r.a_vec))
        log.info('conjugate_gradient: iteration %d/%d, stopping criterion: %.5e',
                 k, max_iter, stopping_factor)

        list_of_stopping_factors.append(stopping_factor)
        if stopping_factor < threshold:
            return a, list_of_stopping_factors
        if stopping_factor < best_stopping_factor:
            best_stopping_factor, best_a = stopping_factor, copy(a)

        if pcond is not None:
            z = pcond.apply_to(r)
        else:
            z = copy(r)
        new_r_dot = mpi_dot_prod(mpi_comm, z.a_vec, r.a_vec)
        p.a_vec = z.a_vec + (new_r_dot / old_r_dot) * p.a_vec

        old_r_dot = new_r_dot


@jit
def compute_rms(signal: Any, samples_per_period: List[int]) -> Any:
    result = np.empty(len(samples_per_period))
    start_idx = 0
    for i, cur_samples in enumerate(samples_per_period):
        subarray = signal[start_idx:start_idx + cur_samples]
        if cur_samples % 2 > 0:
            result[i] = 0.5 * (np.var(subarray[1::2] - subarray[0:-1:2]))
        else:
            result[i] = 0.5 * (np.var(subarray[1::2] - subarray[0::2]))

        start_idx += cur_samples

    return result


class FullPreconditioner:
    def __init__(self, mc: MonopoleAndDipole, pix_idx,
                 samples_per_ofsp, samples_per_gainp):
        assert sum(samples_per_ofsp) == len(pix_idx)

        self.samples_per_ofsp = samples_per_ofsp
        self.samples_per_gainp = samples_per_gainp
        self.ofsp_per_gainp = \
            OfsAndGains.calc_ofsp_per_gainp(
                samples_per_ofsp, samples_per_gainp)

        assert sum(self.ofsp_per_gainp) == len(self.samples_per_ofsp)

        self.matrices = []
        cur_ofsp_idx = 0
        cur_sample_idx = 0
        for ofsp_in_cur_gainp in self.ofsp_per_gainp:
            cur_matrix = np.zeros((ofsp_in_cur_gainp + 1,
                                   ofsp_in_cur_gainp + 1))

            first_sample = cur_sample_idx
            for i, cur_ofsp in enumerate(samples_per_ofsp[cur_ofsp_idx:(cur_ofsp_idx +
                                                                        ofsp_in_cur_gainp)]):
                cur_monopole = mc.monopole_map[pix_idx[cur_sample_idx:(
                    cur_sample_idx + cur_ofsp)]]
                cur_dipole = mc.dipole_map[pix_idx[cur_sample_idx:(
                    cur_sample_idx + cur_ofsp)]]
                cur_matrix[i, i] = np.sum(cur_monopole)
                cur_matrix[ofsp_in_cur_gainp, i] = cur_matrix[i, ofsp_in_cur_gainp] = \
                    np.sum(cur_dipole)

                cur_sample_idx += cur_ofsp

            cur_matrix[ofsp_in_cur_gainp, ofsp_in_cur_gainp] = \
                np.sum(mc.dipole_map[pix_idx[first_sample:cur_sample_idx]]**2)

            # If the determinant is not positive, the matrix is not positive definite!
            assert np.linalg.det(cur_matrix) > 0

            self.matrices.append(np.linalg.inv(cur_matrix))

    def apply_to(self, a: OfsAndGains) -> OfsAndGains:
        cur_ofsp_idx = 0
        cur_gainp_idx = 0
        a_vec = np.empty(len(a.a_vec))
        result = ofs_and_gains_with_same_lengths(a, a_vec)

        gains = a.gains
        offsets = a.offsets
        for cur_gainp_idx, num_of_ofsp in enumerate(self.ofsp_per_gainp):
            # y = M^-1 x   for each block in F^T F
            x = np.empty(num_of_ofsp + 1)
            x[0:num_of_ofsp] = offsets[cur_ofsp_idx:(
                cur_ofsp_idx + num_of_ofsp)]
            x[num_of_ofsp] = gains[cur_gainp_idx]
            y = self.matrices[cur_gainp_idx] @ x

            result.offsets[cur_ofsp_idx:(
                cur_ofsp_idx + num_of_ofsp)] = y[0:num_of_ofsp]
            result.gains[cur_gainp_idx] = y[num_of_ofsp]
            cur_ofsp_idx += num_of_ofsp
            cur_gainp_idx += 1
        return result

    def compute_offset_errors(self, voltages, samples_per_ofsp):
        rms = compute_rms(voltages, samples_per_ofsp)
        return np.sqrt(rms * np.array([np.diag(x)[:-1] for x in self.matrices]).flatten())

    def compute_gain_errors(self, voltages, samples_per_gainp):
        rms = compute_rms(voltages, samples_per_gainp)
        return np.sqrt(rms * np.array([np.diag(x)[-1] for x in self.matrices]))


class JacobiPreconditioner:
    def __init__(self, mc: MonopoleAndDipole, pix_idx,
                 samples_per_ofsp, samples_per_gainp):
        self.diagonal = OfsAndGains(offsets=np.zeros(len(samples_per_ofsp)),
                                    gains=np.zeros(len(samples_per_gainp)),
                                    samples_per_ofsp=samples_per_ofsp,
                                    samples_per_gainp=samples_per_gainp)

        cur_sample_idx = 0
        offsets = self.diagonal.offsets
        for row_idx, cur_sample_num in enumerate(samples_per_ofsp):
            cur_sum = np.sum(mc.monopole_map[pix_idx[cur_sample_idx:(cur_sample_idx +
                                                                     cur_sample_num)]])
            if cur_sum != 0.0:
                offsets[row_idx] = 1.0 / cur_sum
            else:
                offsets[row_idx] = 1.0

            cur_sample_idx += cur_sample_num
        cur_sample_idx = 0
        gains = self.diagonal.gains
        for row_idx, cur_sample_num in enumerate(samples_per_gainp):
            cur_sum = np.sum(mc.dipole_map[pix_idx[cur_sample_idx:(cur_sample_idx +
                                                                   cur_sample_num)]]**2)
            if cur_sum != 0.0:
                gains[row_idx] = 1.0 / cur_sum
            else:
                gains[row_idx] = 1.0

            cur_sample_idx += cur_sample_num

    def apply_to(self, a: OfsAndGains) -> OfsAndGains:
        return ofs_and_gains_with_same_lengths(a, a.a_vec * self.diagonal.a_vec)

    def compute_offset_errors(self, voltages, samples_per_ofsp):
        rms = compute_rms(voltages, samples_per_ofsp)
        return np.sqrt(rms * self.diagonal.offsets)

    def compute_gain_errors(self, voltages, samples_per_gainp):
        rms = compute_rms(voltages, samples_per_gainp)
        return np.sqrt(rms * self.diagonal.gains)


PCOND_DICT = {'none': None,
              'full': FullPreconditioner,
              'jacobi': JacobiPreconditioner}


def guess_gains(voltages, pix_idx, dipole_map, samples_per_gainp):
    log.debug('entering guess_gains')

    result = np.empty(len(samples_per_gainp), dtype='float')
    cal_start = 0
    for gainp_idx, gainp_len in enumerate(samples_per_gainp):
        cal_end = cal_start + gainp_len
        cur_fit = scipy.polyfit(x=dipole_map[pix_idx[cal_start:cal_end]],
                                y=voltages[cal_start:cal_end],
                                deg=1)
        result[gainp_idx] = cur_fit[0]

        cal_start += gainp_len

    return result


DaCapoResults = namedtuple('DaCapoResults',
                           ['ofs_and_gains',
                            'sky_map',
                            'list_of_cg_rz',
                            'list_of_dacapo_rz',
                            'cg_wall_times',
                            'converged',
                            'dacapo_wall_time'])


def da_capo(mpi_comm, voltages, pix_idx, samples_per_ofsp, samples_per_gainp,
            mc: MonopoleAndDipole, mask=None, pcond=None, threshold=1e-9, max_iter=10,
            cg_threshold=1e-9, max_cg_iter=100) -> DaCapoResults:
    log.debug('entering da_capo')

    dacapo_prof = Profiler()

    sky_map = np.zeros_like(mc.dipole_map)
    start_gains = guess_gains(
        voltages, pix_idx, mc.dipole_map, samples_per_gainp)
    old_a = OfsAndGains(offsets=np.zeros(len(samples_per_ofsp)),
                        gains=start_gains,
                        samples_per_ofsp=samples_per_ofsp,
                        samples_per_gainp=samples_per_gainp)

    iteration = 0
    cg_wall_times = []
    list_of_cg_rz = []
    list_of_dacapo_rz = []
    while True:
        log.info('da_capo: iteration %d/%d', iteration + 1, max_iter)

        cg_prof = Profiler()
        new_a, rz = conjugate_gradient(mpi_comm, voltages, old_a, sky_map, pix_idx,
                                       mc, pcond=pcond, threshold=cg_threshold,
                                       max_iter=max_cg_iter)
        list_of_cg_rz.append(rz)
        cg_wall_times.append(cg_prof.toc())
        sky_map_corr = compute_map_corr(mpi_comm, voltages, old_a, new_a,
                                        pix_idx, mc.dipole_map, sky_map)
        sky_map += sky_map_corr

        stopping_factor = mpi_abs_max(mpi_comm, new_a.a_vec - old_a.a_vec)
        list_of_dacapo_rz.append(stopping_factor)
        log.info('da_capo: stopping factor %.3e (threshold is %.3e)',
                 stopping_factor, threshold)

        if stopping_factor < threshold:
            log.info('da_capo: convergence reached after %d steps', iteration)
            return DaCapoResults(ofs_and_gains=new_a,
                                 sky_map=sky_map,
                                 list_of_cg_rz=list_of_cg_rz,
                                 list_of_dacapo_rz=list_of_dacapo_rz,
                                 converged=True,
                                 cg_wall_times=cg_wall_times,
                                 dacapo_wall_time=dacapo_prof.toc())

        old_a = new_a
        iteration += 1

        if iteration >= max_iter:
            log.info('da_capo: maximum number of iterations reached (%d)', max_iter)
            return DaCapoResults(ofs_and_gains=new_a,
                                 sky_map=sky_map,
                                 list_of_cg_rz=list_of_cg_rz,
                                 list_of_dacapo_rz=list_of_dacapo_rz,
                                 converged=False,
                                 cg_wall_times=cg_wall_times,
                                 dacapo_wall_time=dacapo_prof.toc())


def mpi_abs_max(mpi_comm, vec):
    log.debug('entering mpi_abs_max')

    local_max = np.max(np.abs(vec))

    if mpi_comm is not None:
        return mpi_comm.allreduce(local_max, op=MPI.MAX)
    else:
        return local_max


def compute_map_corr(mpi_comm, voltages, old_a: OfsAndGains, new_a: OfsAndGains,
                     pix_idx, dipole_map, sky_map):
    log.debug('entering compute_map_corr')

    diff_tod = voltages - apply_f(new_a, pix_idx, dipole_map, sky_map)
    map_corr = apply_ptildet(mpi_comm, diff_tod, old_a,
                             pix_idx, len(dipole_map))
    normalization = compute_diagm(mpi_comm, old_a, pix_idx, len(dipole_map))
    result = np.ma.array(map_corr, mask=(np.abs(normalization) < 1e-9),
                         fill_value=0.0)
    return (result / normalization).filled()


DEFAULT_LOGFILE_MASK = 'calibrate_%04d.log'


@click.command()
@click.argument('configuration_file')
@click.option('--debug/--no-debug', 'debug_flag',
              help='Print more debugging information during the execution')
@click.option('--full-log/--no-full-log', 'full_log_flag',
              help='Make every MPI process write log message to files'
              ' (use --logfile to specify the file name)')
@click.option('-i', '--index-file', 'indexfile_path', default=None, type=str,
              help='Specify the path to the index file to use '
              '(overrides the one specified in the parameter file).')
@click.option('--logfile', 'logfile_mask', default=DEFAULT_LOGFILE_MASK,
              help='Prints (a subset of) logging messages on the screen'
              ' (default is "{0}")'.format(DEFAULT_LOGFILE_MASK))
def calibrate_main(configuration_file: str, debug_flag: bool,
                   full_log_flag: bool, logfile_mask: str,
                   indexfile_path: str):
    mpi_comm = MPI.COMM_WORLD
    mpi_size = mpi_comm.Get_size()
    mpi_rank = mpi_comm.Get_rank()

    if debug_flag:
        log_level = log.DEBUG
    else:
        log_level = log.INFO

    log_format = '[%(asctime)s %(levelname)s MPI#{0:04d}] %(message)s'.format(
        mpi_rank)
    if full_log_flag:
        log.basicConfig(level=log_level, filename=(logfile_mask % mpi_rank),
                        filemode='w', format=log_format)
    else:
        if mpi_rank == 0:
            log.basicConfig(level=log_level, format=log_format)
        else:
            log.basicConfig(level=log.CRITICAL)
    log.info('reading configuration file "%s"', configuration_file)
    configuration = read_calibrate_conf_file(configuration_file)
    log.info('configuration file read successfully')

    if indexfile_path is not None:
        configuration.index_file = indexfile_path

    if configuration.index_file is None:
        log.error('error: you must specify an index file, either in the '
                  'parameter file or using the --index-file switch')
        sys.exit(1)
    index = IndexFile()
    index.load_from_fits(configuration.index_file,
                         first_idx=configuration.first_tod_index,
                         last_idx=configuration.last_tod_index)
    log.info('%d files are going to be loaded', len(index.tod_info))
    samples_per_ofsp = index.periods

    gainp_lengths = split(length=len(samples_per_ofsp),
                          sublength=configuration.periods_per_cal_constant)
    samples_per_gainp = ftnroutines.sum_subranges(samples_per_ofsp,
                                                  gainp_lengths)

    log.info('number of offset periods: %d; number of gain periods: %d',
             len(samples_per_ofsp), len(gainp_lengths))

    gainp_per_process = split_into_n(length=len(samples_per_gainp),
                                     num_of_segments=mpi_size)
    samples_per_process = ftnroutines.sum_subranges(samples_per_gainp,
                                                    gainp_per_process)

    gainp_idx_start = sum(gainp_per_process[0:mpi_rank])
    gainp_idx_end = gainp_idx_start + gainp_per_process[mpi_rank]

    ofsp_idx_start = sum(gainp_lengths[0:gainp_idx_start])
    ofsp_idx_end = ofsp_idx_start + \
        sum(gainp_lengths[gainp_idx_start:gainp_idx_end])

    local_samples_per_ofsp = samples_per_ofsp[ofsp_idx_start:ofsp_idx_end]
    local_samples_per_gainp = samples_per_gainp[gainp_idx_start:gainp_idx_end]
    files_per_process = assign_files_to_processes(
        samples_per_process, index.tod_info)
    tod = load_tod(files_per_process[mpi_rank], index, configuration)
    assert len(tod.signal) == sum(local_samples_per_ofsp)
    assert len(tod.signal) == sum(local_samples_per_gainp)
    overall_num_of_samples = mpi_comm.allreduce(len(tod.signal), op=MPI.SUM)
    log.info('elements in the TOD: %d (split among %d processes)',
             overall_num_of_samples, mpi_size)
    if configuration.mask_file_path is not None:
        mask = healpy.read_map(configuration.mask_file_path, verbose=False)
        mask = np.array(healpy.ud_grade(
            mask, configuration.nside), dtype='int')
    else:
        mask = None
    directions = healpy.pix2vec(configuration.nside,
                                np.arange(healpy.nside2npix(configuration.nside)))
    dipole_map = get_dipole_temperature(t_cmb_k=configuration.t_cmb_k,
                                        solsys_speed_vec_m_s=configuration.solsys_speed_vec_m_s,
                                        directions=directions,
                                        freq=configuration.frequency_hz)
    mc = MonopoleAndDipole(mask=mask, dipole_map=dipole_map)
    try:
        pcond_class = PCOND_DICT[configuration.pcond]
    except KeyError:
        log.error('Unknown preconditioner "%s", valid choices are: %s',
                  configuration.pcond,
                  ', '.join(['"{0}"'.format(x) for x in PCOND_DICT.keys()]))
        sys.exit(1)

    if pcond_class is not None:
        pcond = pcond_class(mc=mc,
                            pix_idx=tod.pix_idx,
                            samples_per_ofsp=local_samples_per_ofsp,
                            samples_per_gainp=local_samples_per_gainp)
    else:
        pcond = None

    da_capo_results = da_capo(mpi_comm,
                              voltages=tod.signal, pix_idx=tod.pix_idx,
                              samples_per_ofsp=local_samples_per_ofsp,
                              samples_per_gainp=local_samples_per_gainp,
                              mc=mc,
                              mask=mask,
                              threshold=configuration.dacapo_stop,
                              max_iter=configuration.dacapo_maxiter,
                              max_cg_iter=configuration.cg_maxiter,
                              cg_threshold=configuration.cg_stop,
                              pcond=pcond)

    coll_gains = gather_arrays(mpi_comm, da_capo_results.ofs_and_gains.gains)
    coll_offsets = gather_arrays(
        mpi_comm, da_capo_results.ofs_and_gains.offsets)
    if pcond is not None:
        coll_gain_errors = gather_arrays(mpi_comm,
                                         pcond.compute_gain_errors(tod.signal,
                                                                   local_samples_per_gainp))

        coll_offset_errors = gather_arrays(mpi_comm,
                                           pcond.compute_offset_errors(tod.signal,
                                                                       local_samples_per_ofsp))

    else:
        log.warning(
            'no preconditioner used, all offset/gain errors will be set to zero')
        coll_offset_errors = np.zeros_like(coll_offsets)
        coll_gain_errors = np.zeros_like(coll_gains)
    if mpi_rank == 0:
        primary_hdu = fits.PrimaryHDU(
            data=configuration.parameter_file_contents)
        primary_hdu.header.add_comment(configuration.comment)
        primary_hdu.header['WTIME'] = (da_capo_results.dacapo_wall_time,
                                       'Wall clock time [s]')
        primary_hdu.header['MPIPROC'] = (mpi_comm.Get_size(),
                                         'Number of MPI processes used')
        primary_hdu.header['CONVERG'] = (da_capo_results.converged,
                                         'Has the DaCapo algorithm converged?')
        primary_hdu.header['GPPEROP'] = (configuration.periods_per_cal_constant,
                                         'Number of ofs periods per each gain period')
        primary_hdu.header['CGSTOP'] = (
            configuration.cg_stop, 'Stopping factor for CG')
        primary_hdu.header['CGMAXIT'] = (
            configuration.cg_maxiter, 'Maximum number of CG iterations')
        primary_hdu.header['DCSTOP'] = (
            configuration.dacapo_stop, 'Stopping factor for DaCapo')
        primary_hdu.header['DCMAXIT'] = (
            configuration.dacapo_maxiter, 'Maximum number of DaCapo iterations')
        primary_hdu.header['PCOND'] = (
            configuration.pcond, 'Kind of preconditioner')
        primary_hdu.header['NSIDE'] = (
            configuration.nside, 'Resolution of the map used by DaCapo')

        if mask is None:
            fsky = 100.0
        else:
            fsky = len(mask[mask > 0]) * 100.0 / len(mask)
        primary_hdu.header['FSKY'] = (
            fsky, 'Fraction of the sky used by DaCapo')
        hdu_list = [primary_hdu] + index.store_in_hdus()
        cols = [fits.Column(name='OFFSET', array=np.array(coll_offsets).flatten(), format='1D'),
                fits.Column(name='ERR', array=np.array(
                    coll_offset_errors).flatten(), format='1D'),
                fits.Column(name='NSAMPLES', array=samples_per_ofsp, format='1J')]
        hdu_list.append(fits.BinTableHDU.from_columns(cols, name='OFFSETS'))

        cols = [fits.Column(name='GAIN', array=np.array(coll_gains).flatten(), format='1D'),
                fits.Column(name='ERR', array=np.array(
                    coll_gain_errors).flatten(), format='1D'),
                fits.Column(name='NSAMPLES', array=samples_per_gainp, format='1J')]
        hdu_list.append(fits.BinTableHDU.from_columns(cols, name='GAINS'))
        if configuration.save_map:
            col = fits.Column(name='SIGNAL', format='D',
                              array=da_capo_results.sky_map)
            hdu = fits.BinTableHDU.from_columns([col], name='SKYMAP')

            hdu.header['PIXTYPE'] = ('HEALPIX', 'HEALPIX pixelisation')
            hdu.header['ORDERING'] = (
                'RING', 'Pixel ordering scheme, either RING or NESTED')
            hdu.header['NSIDE'] = (configuration.nside,
                                   'Healpix''s resolution parameter')
            hdu.header['FIRSTPIX'] = (0, 'First pixel # (0-based)')
            hdu.header['LASTPIX'] = (healpy.nside2npix(configuration.nside) - 1,
                                     'Last pixel # (0-based)')
            hdu.header['INDXSCHM'] = (
                'IMPLICIT', 'Indexing: IMPLICIT or EXPLICIT')
            hdu.header['OBJECT'] = (
                'FULLSKY', 'Sky coverage, either PARTIAL or FULLSKY')

            hdu_list.append(hdu)
        if configuration.save_convergence is not None:
            for idx, cur_cg_rz_list, cur_dacapo_rz, cg_wall_time \
                in zip(range(len(da_capo_results.list_of_dacapo_rz)),
                       da_capo_results.list_of_cg_rz,
                       da_capo_results.list_of_dacapo_rz,
                       da_capo_results.cg_wall_times):

                col = fits.Column(name='RZ', array=cur_cg_rz_list, format='1D')
                hdu = fits.BinTableHDU.from_columns(
                    [col], name='RZ{0:04d}'.format(idx))
                hdu.header['DACAPORZ'] = (
                    cur_dacapo_rz, 'DaCapo stopping factor')
                hdu.header['WTIME'] = (cg_wall_time, 'CG wall clock time [s]')
                hdu_list.append(hdu)
        fits.HDUList(hdu_list).writeto(configuration.output_file_name,
                                       overwrite=True)
        log.info('gains and offsets written into file "%s"',
                 configuration.output_file_name)


if __name__ == '__main__':
    calibrate_main()
