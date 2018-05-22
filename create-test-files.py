#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''Create a set of FITS files to be used as test input for the DaCapo
calibration codes.
'''

import logging as log
import os.path
from typing import Any

from astropy.io import fits
import numpy as np
import healpy
import click
from calibrate import get_dipole_temperature
import quaternionarray as qa


def create_dipole_map(nside: int):
    '''Create a Healpix map containing the CMB dipole signal in K_CMB'''
    t_cmb_k = 2.7
    solsys_colat_rad = 1.7656131194951572
    solsysdir_long_rad = 2.995889600573578
    solsysspeed_m_s = 370082.2332
    solsys_speed_vec_m_s = solsysspeed_m_s * \
        np.array([np.sin(solsys_colat_rad) * np.cos(solsysdir_long_rad),
                  np.sin(solsys_colat_rad) * np.sin(solsysdir_long_rad),
                  np.cos(solsys_colat_rad)])
    return get_dipole_temperature(
        t_cmb_k=t_cmb_k,
        solsys_speed_vec_m_s=solsys_speed_vec_m_s,
        directions=healpy.pix2vec(nside, np.arange(healpy.nside2npix(nside))))


def write_simulated_tod(file_name: str, time: Any, theta: Any, phi: Any, tod: Any,
                        offsets: Any, gains, samples_per_ofsp: int, samples_per_gainp:
                        int):
    hdu1 = fits.BinTableHDU.from_columns([
        fits.Column(name='TIME', array=time, format='J', unit='s'),
        fits.Column(name='THETA', array=theta, format='E', unit='rad'),
        fits.Column(name='PHI', array=phi, format='E', unit='rad'),
        fits.Column(name='SIGNAL', array=tod, format='D', unit='V')])

    hdu2 = fits.BinTableHDU.from_columns([
        fits.Column(name='OFFSET', array=offsets, format='D', unit='V')],
        name='OFFSETS')
    hdu2.header['OFSSAMP'] = (samples_per_ofsp, 'Samples per offset period')

    hdu3 = fits.BinTableHDU.from_columns([
        fits.Column(name='GAIN', array=gains, format='D', unit='V/K')],
        name='GAINS')
    hdu3.header['GAINSAMP'] = (samples_per_gainp, 'Samples per gain period')
    hdu3.header['GAINSOFS'] = (samples_per_gainp // samples_per_ofsp,
                               'Offset periods per gain period')

    hdulist = fits.HDUList([fits.PrimaryHDU(), hdu1, hdu2, hdu3])
    hdulist.writeto(file_name, clobber=True)


def generate_pointings(rpm1: float, rpm2: float, times: Any, num_of_samples:
                       int):
    quat1 = np.tile(
        qa.rotation([1., 0, 0], np.pi / 2.), num_of_samples).reshape(-1, 4)
    quat2 = qa.rotation([0, 0, 1.], 2. * np.pi * rpm1 / 60.0 * times)
    quat3 = np.tile(
        qa.rotation([1., 0, 0], np.pi / 3.), num_of_samples).reshape(-1, 4)
    quat4 = qa.rotation([0, 0, 1.], 2. * np.pi * rpm2 / 60.0 * times)

    fullquat = qa.mult(quat4, qa.mult(quat3, qa.mult(quat2, quat1)))
    vectors = qa.rotate(fullquat, np.array([0., 0., 1.]))
    return healpy.vec2ang(vectors[:, 0:3])


def decalibrate_tod(tod, offsets, gains, samples_per_ofsp, samples_per_gainp):
    start_idx = 0
    cur_ofsp = 0
    cur_gainp = 0
    samples_in_gainp = 0
    while start_idx < len(tod):
        tod[start_idx:start_idx + samples_per_ofsp] = \
            tod[start_idx:start_idx + samples_per_ofsp] * gains[cur_gainp] + \
            offsets[cur_ofsp]

        start_idx += samples_per_ofsp
        cur_ofsp += 1
        samples_in_gainp += samples_per_ofsp
        if samples_in_gainp >= samples_per_gainp:
            cur_gainp += 1
            samples_in_gainp = 0


@click.command()
@click.argument('outdir',
                type=click.Path(exists=True,
                                dir_okay=True,
                                file_okay=False))
def main(outdir: str):
    log.basicConfig(level=log.INFO,
                    format='[%(asctime)s %(levelname)s] %(message)s')

    samples_per_ofsp = 5000  # Number of samples
    samples_per_gainp = samples_per_ofsp * 10
    num_of_samples = samples_per_gainp * 20

    assert num_of_samples % samples_per_ofsp == 0
    assert num_of_samples % samples_per_gainp == 0
    assert samples_per_gainp % samples_per_ofsp == 0
    galaxy_map = healpy.read_map(
        os.path.join('maps', 'COM_CMB_IQU-commander_256_ecliptic.fits'),
        verbose=False)
    nside = healpy.npix2nside(len(galaxy_map))

    dipole_map = create_dipole_map(nside=nside)
    dipole_amplitude = np.max(dipole_map) - np.min(dipole_map)
    times = np.linspace(0, 86400., num_of_samples)
    theta, phi = generate_pointings(rpm1=1.,
                                    rpm2=1. / (24. * 60.),
                                    times=times,
                                    num_of_samples=num_of_samples)
    pixidx = healpy.ang2pix(nside, theta, phi)
    healpy.write_map(os.path.join(outdir, 'long_test_hits.fits.gz'),
                     np.bincount(pixidx, minlength=healpy.nside2npix(nside)),
                     overwrite=True)
    tod = (galaxy_map[pixidx] + dipole_map[pixidx] + np.random.randn() *
           dipole_amplitude * 1e-5)
    offsets = np.random.randn(num_of_samples //
                              samples_per_ofsp) * np.sqrt(np.var(dipole_map))
    gains = (np.random.randn(num_of_samples // samples_per_gainp) + 50.0)
    decalibrate_tod(tod, offsets, gains, samples_per_ofsp, samples_per_gainp)
    tod_file_name = 'long_test_tod.fits'
    write_simulated_tod(file_name=os.path.join(outdir, tod_file_name),
                        time=times,
                        theta=theta,
                        phi=phi,
                        tod=tod,
                        offsets=offsets,
                        gains=gains,
                        samples_per_ofsp=samples_per_ofsp,
                        samples_per_gainp=samples_per_gainp)
    log.info('file "%s" written', tod_file_name)

    index_file_name = os.path.join(outdir, 'long_test_index.fits')
    with open(os.path.join(outdir, 'long_test_index.ini'), 'wt') as f:
        f.write('''[input_files]
    path = {path}
    mask = {tod_file_name}
    hdu = 1
    column = TIME

    [periods]
    length = {samples_per_ofsp}

    [output_file]
    file_name = {index_file_name}
    '''.format(path=outdir,
               tod_file_name=tod_file_name,
               index_file_name=index_file_name,
               samples_per_ofsp=times[samples_per_ofsp]))
    for pcond in ('none', 'jacobi', 'full'):
        ini_file_name = os.path.join(
            outdir, 'long_test_calibrate_{0}.ini'.format(pcond))
        output_file_name = os.path.join(
            outdir, 'long_test_results_{0}.fits'.format(pcond))

        with open(ini_file_name, 'wt') as f:
            f.write('''[input_files]
    index_file = {index_file_name}
    signal_hdu = 1
    signal_column = SIGNAL
    pointing_hdu = 1
    pointing_columns = THETA, PHI

    [dacapo]
    t_cmb_K = 2.72548
    solsysdir_ecl_colat_rad = 1.7656131194951572
    solsysdir_ecl_long_rad = 2.995889600573578
    solsysspeed_m_s = 370082.2332
    nside = {nside}
    periods_per_cal_constant = {gainp_per_ofsp}
    cg_stop_value = 1e-9
    cg_max_iterations = 100
    dacapo_stop_value = 1e-9
    dacapo_max_iterations = 20
    pcond = {pcond}

    [output]
    file_name = {output_file_name}
    save_map = yes
    save_convergence = yes
    comment = "Long duration test"
    '''.format(index_file_name=index_file_name,
               gainp_per_ofsp=samples_per_gainp // samples_per_ofsp,
               output_file_name=output_file_name,
               pcond=pcond,
               nside=nside))

    log.info('file "%s" written', ini_file_name)


if __name__ == '__main__':
    main()
