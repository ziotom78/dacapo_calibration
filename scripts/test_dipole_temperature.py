import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import healpy
import matplotlib.pylab as plt
import numpy as np
from calibrate import get_dipole_temperature
nside = 64
solsys_speed_vec_m_s = np.array([-359234.3, 52715.1, -71643.0])
directions = healpy.pix2vec(nside, np.arange(healpy.nside2npix(nside)))
dipole_map = get_dipole_temperature(t_cmb_k=2.72548,
                                    solsys_speed_vec_m_s=solsys_speed_vec_m_s,
                                    directions=directions)
healpy.mollview(dipole_map, title='', unit='K', cmap='gray_r')
plt.savefig('test_dipole_temperature.pdf', bbox_inches='tight')
