# -*- encoding: utf-8 -*-

import unittest as ut

import numpy as np
from calibrate import OfsAndGains, MonopoleAndDipole, \
    apply_f, apply_ft, compute_diagm, apply_ptilde, apply_ptildet, \
    apply_z, apply_A, compute_v, conjugate_gradient, compute_map_corr, \
    da_capo, JacobiPreconditioner, FullPreconditioner, compute_rms


def check_vector_match(name, vector1, vector2, rtol=1e-05):
    try:
        assert np.allclose(vector1, vector2, rtol=rtol)
        print('test "{0}" passed'.format(name))

    except:
        # This catches both failed assertions and mismatches in the
        # shapes of vec1/vec2
        print('ERROR: test "{0}" failed'.format(name))
        print('       vectors {0} and {1} are not the same'
              .format(vector1, vector2))
        raise


def build_test_matrices(num_of_samples, num_of_pixels, pix_idx,
                        mc, signal_sum_map, ofs_and_gains):
    P = np.zeros((len(pix_idx), num_of_pixels), dtype='int')
    for i, pixel in enumerate(pix_idx):
        P[i][pixel] = 1
    F = np.zeros((num_of_samples, len(
        ofs_and_gains.offsets) + len(ofs_and_gains.gains)))
    start_idx = 0
    for col, ofs_len in enumerate(ofs_and_gains.samples_per_ofsp):
        F[start_idx:(start_idx + ofs_len), col] = 1
        start_idx += ofs_len

    start_idx = 0
    for col, gain_len in enumerate(ofs_and_gains.samples_per_gainp):
        F[start_idx:(start_idx + gain_len), len(ofs_and_gains.offsets) + col] = \
            signal_sum_map[pix_idx[start_idx:(start_idx + gain_len)]]
        start_idx += gain_len
    Gext = np.repeat(ofs_and_gains.gains, ofs_and_gains.samples_per_gainp)
    ptilde = np.array([P[i] * Gext[i] for i in range(len(Gext))])
    M = ptilde.T @ ptilde
    invM = np.linalg.inv(M)
    mat2x2 = mc.T @ invM @ mc
    MCminv = invM - invM @ mc @ np.linalg.inv(mat2x2) @ mc.T @ invM
    Z = np.eye(num_of_samples) - ptilde @ MCminv @ ptilde.T
    A = F.T @ Z @ F

    return P, F, ptilde, M, Z, A


class TestDaCapo(ut.TestCase):

    def setUp(self):
        self.num_of_pixels = 3
        self.sky_map = np.array([-0.4, 0.2, 0.2])
        self.D = np.sin(2 * np.pi * np.array([0, 1 / 3, 2 / 3]))
        self.mc = np.array([self.D, [1, 1, 1]]).T
        self.mon_and_dip = MonopoleAndDipole(mask=[1, 1, 1], dipole_map=self.D)
        self.signal_sum_map = self.D + self.sky_map
        self.pix_idx = np.array([0, 0, 1, 0, 1, 2, 2, 2, 0, 1, 0])
        self.ofs_and_gains = OfsAndGains(offsets=np.array([10.0, 10.5]),
                                         gains=np.array([4.1, 4.2]),
                                         samples_per_ofsp=[6, 5],
                                         samples_per_gainp=[6, 5])

        self.a_vec = self.ofs_and_gains.a_vec
        self.Gext = np.repeat(self.ofs_and_gains.gains,
                              self.ofs_and_gains.samples_per_gainp)
        self.bext = np.repeat(self.ofs_and_gains.offsets,
                              self.ofs_and_gains.samples_per_ofsp)
        self.P, self.F, self.ptilde, self.M, self.Z, self.A = \
            build_test_matrices(len(self.bext), self.num_of_pixels, self.pix_idx,
                                self.mc, self.signal_sum_map, self.ofs_and_gains)
        self.tod = self.Gext * (self.P @ self.signal_sum_map) + self.bext

    def testApplyF(self):
        check_vector_match('apply_f',
                           self.F @ self.a_vec,
                           apply_f(self.ofs_and_gains, self.pix_idx,
                                   self.D, self.sky_map))

    def testApplyFT(self):
        check_vector_match('apply_ft',
                           self.F.T @ self.tod,
                           apply_ft(self.tod, self.ofs_and_gains, self.pix_idx,
                                    self.D, self.sky_map))

    def testComputeDiagM(self):
        check_vector_match('compute_diagm',
                           np.diag(self.M),
                           compute_diagm(None, self.ofs_and_gains,
                                         self.pix_idx, self.num_of_pixels))

    def testApplyPtilde(self):
        check_vector_match('apply_ptilde',
                           self.ptilde @ self.sky_map,
                           apply_ptilde(self.sky_map, self.ofs_and_gains,
                                        self.pix_idx))

    def testApplyPtildet(self):
        check_vector_match('apply_ptildet',
                           self.ptilde.T @ self.tod,
                           apply_ptildet(None, self.tod, self.ofs_and_gains,
                                         self.pix_idx, self.num_of_pixels))

    def testApplyZ(self):
        check_vector_match('apply_z',
                           self.Z @ self.tod,
                           apply_z(None, self.tod, self.ofs_and_gains,
                                   self.pix_idx, self.mon_and_dip))

    def testApplyA(self):
        ofs_gains_guess = OfsAndGains(offsets=np.zeros(2),
                                      gains=np.ones(2),
                                      samples_per_ofsp=self.ofs_and_gains.samples_per_ofsp,
                                      samples_per_gainp=self.ofs_and_gains.samples_per_gainp)
        check_vector_match('apply_A',
                           self.A @ ofs_gains_guess.a_vec,
                           apply_A(None, self.ofs_and_gains, self.sky_map,
                                   self.pix_idx, self.mon_and_dip,
                                   ofs_gains_guess))

    def testComputeV(self):
        check_vector_match('compute_v',
                           self.F.T @ self.Z @ self.tod,
                           compute_v(None, self.tod, self.ofs_and_gains, self.sky_map,
                                     self.pix_idx, self.mon_and_dip))

    def testComputeMapCorr(self):
        fake_tod = np.arange(len(self.tod))
        check_vector_match('compute_map_corr',
                           np.linalg.inv(self.ptilde.T @ self.ptilde) @
                           self.ptilde.T @ (fake_tod - self.F @ self.a_vec),
                           compute_map_corr(None, fake_tod, self.ofs_and_gains,
                                            self.ofs_and_gains,
                                            self.pix_idx, self.D, self.sky_map))

    def precondition_check(self, precObj):
        ofs_gains_guess = OfsAndGains(offsets=np.zeros(2),
                                      gains=np.ones(2),
                                      samples_per_ofsp=self.ofs_and_gains.samples_per_ofsp,
                                      samples_per_gainp=self.ofs_and_gains.samples_per_gainp)

        P, F, ptilde, M, Z, A = build_test_matrices(len(self.bext),
                                                    self.num_of_pixels, self.pix_idx,
                                                    self.mc, self.D,
                                                    ofs_gains_guess)

        if precObj:
            pcond = precObj(mc=self.mon_and_dip,
                            pix_idx=self.pix_idx,
                            samples_per_ofsp=self.ofs_and_gains.samples_per_ofsp,
                            samples_per_gainp=self.ofs_and_gains.samples_per_gainp)
        else:
            pcond = None

        cg_a, _ = conjugate_gradient(None, self.tod, ofs_gains_guess,
                                     np.zeros_like(self.D),
                                     self.pix_idx, self.mon_and_dip, pcond=pcond)
        check_vector_match('conjugate_gradient',
                           np.linalg.inv(A) @ F.T @ Z @ self.tod, cg_a.a_vec)

        result = da_capo(mpi_comm=None,
                         voltages=self.tod,
                         pix_idx=self.pix_idx,
                         samples_per_ofsp=self.ofs_and_gains.samples_per_ofsp,
                         samples_per_gainp=self.ofs_and_gains.samples_per_gainp,
                         mc=self.mon_and_dip,
                         pcond=pcond)
        check_vector_match('da_capo (offsets)',
                           self.ofs_and_gains.offsets, result.ofs_and_gains.offsets)
        check_vector_match('da_capo (gains)',
                           self.ofs_and_gains.gains, result.ofs_and_gains.gains)
        check_vector_match('da_capo (map)',
                           self.sky_map, result.sky_map)

    def testNoPreconditioner(self):
        self.precondition_check(None)

    def testJacobiPreconditioner(self):
        self.precondition_check(JacobiPreconditioner)

    def testFullPreconditioner(self):
        self.precondition_check(FullPreconditioner)


class TestMiscellanea(ut.TestCase):

    def testComputeRMS(self):
        chunks = [40000, 70000, 50000]
        samples = np.random.randn(np.sum(chunks))
        check_vector_match('compute_rms',
                           np.repeat(1.0, len(chunks)),
                           compute_rms(samples, chunks),
                           rtol=5e-2)
