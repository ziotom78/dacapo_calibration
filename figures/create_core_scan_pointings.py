#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

'''To run this program, you need TOAST:

http://hpc4cmb.github.io/toast/

It is used to create core_scanning.fits, which is used by "core_scan_plot.py"
to produce the figure included in the main document.
'''

import logging as log

import astropy.io.fits as fits
import healpy
import numpy as np  # typing: ignore
from dipole import get_dipole_temperature
import toast
import toast.tod as tt

__version__ = '0.0.1'

DETECTOR_NAME = 'boresight'
DEG_ANGLE_PER_DAY = 360.0 / 365.25
SAMPLE_RATE = 10.0
NUM_OF_HOURS = 0.5
ONEOVERF_KNEE_HZ = 0.020
ONEOVERF_ALPHA = 1.0
NOISE_NET = 5.23e-5
SPIN_PERIOD_MIN = 2.0
SPIN_ANGLE_DEG = 65
PREC_PERIOD_MIN = 5760
PREC_ANGLE_DEG = 30
GAIN = 35.0

################################################################################

def main():
    'The main function'

    # Set up the logging system
    logformat = '[%(asctime)s %(levelname)s] %(message)s'
    log.basicConfig(level=log.INFO,
                    format=logformat)

    # This is the 2-level toast communicator.  By default,
    # there is just one group which spans MPI_COMM_WORLD.
    comm = toast.Comm()

    if comm.comm_world.rank == 0:
        log.info("running with %d processes", comm.comm_world.size)

    # construct the list of intervals which split the whole observation time
    intervals = tt.regular_intervals(n=1,
                                     start=0.0,
                                     first=0,
                                     rate=SAMPLE_RATE,
                                     duration=3600*float(NUM_OF_HOURS),
                                     gap=0)

    # Since we are using a single observation (for madam compatibility),
    # the observation may be very long.  Each interval within the observation
    # may also be long (e.g. a day).  Sub-divide each interval into
    # smaller pieces for purposes of data distribution.

    detquats = {}  # type: Dict[str, Any]
    detquats[DETECTOR_NAME] = np.array([0.0, 0.0, 1.0, 0.0])

    # Create the noise model for this observation

    rate = {}  # type: Dict[str, float]
    fmin = {}  # type: Dict[str, float]
    fknee = {}  # type: Dict[str, float]
    alpha = {}  # type: Dict[str, float]
    net = {}  # type: Dict[str, float]
    rate[DETECTOR_NAME] = SAMPLE_RATE
    fknee[DETECTOR_NAME] = ONEOVERF_KNEE_HZ
    fmin[DETECTOR_NAME] = 1.0/ (NUM_OF_HOURS * 3600.0)
    alpha[DETECTOR_NAME] = ONEOVERF_ALPHA
    net[DETECTOR_NAME] = NOISE_NET

    noise = tt.AnalyticNoise(rate=rate,
                             fmin=fmin,
                             detectors=[DETECTOR_NAME],
                             fknee=fknee,
                             alpha=alpha,
                             NET=net)

    # The distributed timestream data
    data = toast.Data(comm)

    intsamp = None
    if len(intervals) == 1:
        intsamp = intervals[0].last - intervals[0].first + 1
    else:
        # include also the samples in the gap
        intsamp = intervals[1].first - intervals[0].first

    distint = [intsamp]
    distsizes = []
    for _ in intervals:
        distsizes.extend(distint)

    totsamples = np.sum(distsizes)

    # create the single TOD for this observation
    tod = tt.TODSatellite(
        mpicomm=comm.comm_group,
        detectors=detquats,
        samples=totsamples,
        firsttime=0.0,
        rate=SAMPLE_RATE,
        spinperiod=SPIN_PERIOD_MIN,
        spinangle=SPIN_ANGLE_DEG,
        precperiod=PREC_PERIOD_MIN,
        precangle=PREC_ANGLE_DEG,
        sizes=distsizes
    )

    # Create the (single) observation

    ob = {} # type: Dict[str, Any]
    ob['id'] = 0
    ob['tod'] = tod
    ob['intervals'] = intervals
    ob['noise'] = noise

    data.obs.append(ob)

    precquat = tt.slew_precession_axis(nsim=tod.local_samples[1],
                                       firstsamp=tod.local_samples[0],
                                       samplerate=SAMPLE_RATE,
                                       degday=DEG_ANGLE_PER_DAY)

    # we set the precession axis now, which will trigger calculation
    # of the boresight pointing.
    tod.set_prec_axis(qprec=precquat)

    # simulate noise

    nse = tt.OpSimNoise()
    nse.exec(data)

    comm.comm_world.barrier()

    # make a Healpix pointing matrix.  By setting purge_pntg=True,
    # we purge the detector quaternion pointing to save memory.
    # If we ever change this pipeline in a way that needs this
    # pointing at a later stage, we need to set this to False
    # and run at higher concurrency.

    pointing = tt.OpPointingHpix(nside=1024,
                                 nest=True,
                                 mode='IQU',
                                 hwprpm=0.0,
                                 hwpstep=None,
                                 hwpsteptime=None)
    pointing.exec(data)

    foreground_map = healpy.read_map('../foreground_maps/70GHz_ecliptic.fits', verbose=False)

    times = tod.read_times()
    flags, glflags = tod.read_flags(detector=DETECTOR_NAME)

    noise = data.obs[0]['tod'].cache.reference('noise_{0}'.format(DETECTOR_NAME))
    quaternions = tod.read_pntg(detector=DETECTOR_NAME)
    theta, phi, psi = tt.quat2angle(quaternions)
    vect = healpy.ang2vec(theta, phi)
    foreground_tod, dipole_tod = (healpy.get_interp_val(foreground_map,
                                                        theta, phi),
                                  get_dipole_temperature(np.column_stack(vect)))
    total_tod = GAIN * (foreground_tod + dipole_tod + noise)
    hdu = fits.BinTableHDU.from_columns([fits.Column(name='TIME',
                                                     format='D',
                                                     unit='s',
                                                     array=times),
                                         fits.Column(name='THETA',
                                                     format='E',
                                                     unit='rad',
                                                     array=theta),
                                         fits.Column(name='PHI',
                                                     format='E',
                                                     unit='rad',
                                                     array=phi),
                                         fits.Column(name='PSI',
                                                     format='E',
                                                     unit='rad',
                                                     array=psi),
                                         fits.Column(name='FGTOD',
                                                     format='E',
                                                     unit='K',
                                                     array=foreground_tod),
                                         fits.Column(name='DIPTOD',
                                                     format='E',
                                                     unit='K',
                                                     array=dipole_tod),
                                         fits.Column(name='NOISEPSD',
                                                     format='E',
                                                     unit='K',
                                                     array=noise),
                                         fits.Column(name='TOTALTOD',
                                                     format='E',
                                                     unit='V',
                                                     array=total_tod),
                                         fits.Column(name='FLAGS',
                                                     format='I',
                                                     unit='',
                                                     array=flags),
                                         fits.Column(name='GLFLAGS',
                                                     format='I',
                                                     unit='',
                                                     array=glflags)])
    hdu.header['COMMENT'] = 'Angles are expressed in the Ecliptic system'
    hdu.header['FIRSTT'] = (times[0], 'Time of the first sample [s]')
    hdu.header['LASTT'] = (times[-1], 'Time of the last sample [s]')
    hdu.header['GAIN'] = (GAIN, 'Detector gain [V/K]')
    hdu.header['NAME'] = (DETECTOR_NAME, 'Name of the detector')
    hdu.header['VERSION'] = (__version__,
                             'Version of the code used to create '
                             'this file')

    hdu.header['net'] = (NOISE_NET, 'net')
    hdu.header['ALPHA'] = (ONEOVERF_ALPHA, 'Slope of the 1/f noise')
    hdu.header['FKNEE'] = (ONEOVERF_KNEE_HZ,
                           'Knee frequency of the 1/f noise [Hz]')

    hdu.writeto('core_scan_example.fits', clobber=True)

##############################################################################

if __name__ == "__main__":
    main()
