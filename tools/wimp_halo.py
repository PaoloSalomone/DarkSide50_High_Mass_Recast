import numericalunits as nu
import numpy as np
import pandas as pd
from scipy.special import erf
from scipy import stats, integrate
import matplotlib.pyplot as plt

class WimpHalo:

  def __init__(self): 
    nu.reset_units('SI')    
    #Most probable velocity of WIMPs in the halo,relative to galactic center (asymptotic)
    self.v_0    = 220 * nu.km/nu.s
    
    #Local dark matter density
    self.rho_dm = 0.3 * nu.GeV/nu.c0**2 / nu.cm**3
    
    #Galactic escape velocity"""
    self.v_esc  = 544 * nu.km/nu.s

  def j2000(self, year=None, month=None, day_of_month=None, date=None):
      """Convert calendar date in year, month (starting at 1) and
      the (possibly fractional) day of the month relative to midnight UT.
      Either pass year, month and day_of_month or pass pandas datetime object
      via date argument.
      Returns the fractional number of days since J2000.0 epoch.
      """
      if date is not None:
          year = date.year
          month = date.month

          start_of_month = pd.datetime(year, month, 1)
          day_of_month = (date - start_of_month) / pd.Timedelta(1, 'D') + 1

      assert month > 0
      assert month < 13

      y = year if month > 2 else year - 1
      m = month if month > 2 else month + 12

      return (np.floor(365.25 * y)
              + np.floor(30.61 * (m + 1))
              + day_of_month - 730563.5)
              
  def earth_velocity(self, t):
      """Returns 3d velocity of earth, in the galactic rest frame,
      in galactic coordinates.
      :param t: J2000.0 timestamp

      Values and formula from https://arxiv.org/abs/1209.3339
      Assumes earth circular orbit.
      """
      # e_1 and e_2 are the directions of earth's velocity at t1
      # and t1 + 0.25 year.
      e_1 = np.array([0.9931, 0.1170, -0.01032])
      e_2 = np.array([-0.0670, 0.4927, -0.8676])
      # t1 is the time of the vernal equinox, March 21. Does it matter what
      # year? Precession of equinox takes 25800 years so small effect.
      t1 = self.j2000(2000, 3, 21)
      # Angular frequency
      omega = 2 * np.pi / 365.25
      phi = omega * (t - t1)

      # Mean orbital velocity of the Earth (Lewin & Smith appendix B)
      v_orbit = 29.79 * nu.km / nu.s

      v_earth_sun = v_orbit * (e_1 * np.cos(phi) + e_2 * np.sin(phi))

      # Velocity of Local Standard of Rest
      v_lsr = np.array([0, 220, 0]) * nu.km/nu.s
      # Solar peculiar velocity
      v_pec = np.array([11, 12, 7]) * nu.km/nu.s

      return v_lsr + v_pec + v_earth_sun
  
  def v_earth(self,t=None):
      """Return speed of earth relative to galactic rest frame
      :param t: J2000 timestamp or None
      """
      if t is None:
          # Velocity of earth/sun relative to gal. center
          # (eccentric orbit, so not equal to v_0)
          return 232 * nu.km / nu.s
      else:
          return np.sum(self.earth_velocity(t) ** 2) ** 0.5

  def v_max(self, t=None):
      """Return maximum observable dark matter velocity on Earth."""
      if t is None:
          return self.v_esc + self.v_earth(t)
      else:
          return self.v_esc + np.sum(self.earth_velocity(t) ** 2) ** 0.5

  
  def observed_speed_dist(self, v, t=None):
      """Observed distribution of dark matter particle speeds on earth
      under the standard halo model.

      Optionally supply J2000.0 time t to take into account Earth's orbital
      velocity.
      """
      v_earth_t = self.v_earth(t)

      # Normalization constant, see Lewin&Smith appendix 1a
      _w = self.v_esc/self.v_0
      k = erf(_w) - 2/np.pi**0.5 * _w * np.exp(-_w**2)

      # Maximum cos(angle) for this velocity, otherwise v0
      xmax = np.minimum(1,
                        (self.v_esc**2 - v_earth_t**2 - v**2)
                        / (2 * v_earth_t * v))

      y = (k * v / (np.pi**0.5 * self.v_0 * v_earth_t)
           * (np.exp(-((v-v_earth_t)/self.v_0)**2)
              - np.exp(-1/self.v_0**2 * (v**2 + v_earth_t**2
                                      + 2 * v * v_earth_t * xmax))))

      # Zero if v > v_max
      try:
          len(v)
      except TypeError:
          # Scalar argument
          if v > self.v_max(t):
              return 0
          else:
              return y
      else:
          # Array argument
          y[v > self.v_max(t)] = 0
          return y

  def integral_mean_inverse_speed(self, v_min, t=None):
    zz = self.v_esc/self.v_0
    N_esc = erf(zz) - 2.0 * zz / np.pi**0.5 * np.exp(-zz**2.)
    xx = v_min/self.v_0
    yy = self.v_earth(t)/self.v_0

    if zz < yy and xx < np.abs(yy-zz):
      return 1 / self.v_0 / yy
    if zz > yy and xx < np.abs(yy-zz):
      return 1 / (2*N_esc*self.v_0*yy) * (erf(xx+yy) - erf(xx-yy) - 4*np.pi**(-0.5)*yy*np.exp(-zz**2.))
    if np.abs(yy-zz) < xx and xx < (yy+zz):
      return 1 / (2*N_esc*self.v_0*yy) * (erf(zz) - erf(xx-yy) - 2*np.pi**(-0.5)*(yy+zz-xx)*np.exp(-zz**2.))
    return 0

  def galactic_v_dist(self, v):
    # Maxwell Boltzmann distribution with
    # 1/2 m v0**2 = k T <=> v0**2 = 2 k T / m = 2 a**2 <=> a = v_0/sqrt(2)
    # Cut off above escape velocity (and renormalized)
    # See Donato et al, https://arxiv.org/pdf/hep-ph/9803295.pdf, eq. 4/5
    dist = stats.maxwell(scale=self.v_0/2**0.5)
    y = dist.pdf(v) / dist.cdf(self.v_esc)
    if isinstance(v, np.ndarray):
        y[v > self.v_esc] = 0
    elif v > self.v_esc:
         return 0
    return y
  
  
  def mean_inv_speed(self, vmin, t=None):
    """Mean inverse speed function of incoming dark matter.
       See Eqn 1 in arXiv: 1802.06998 and below Eqn 5 in arXiv: 1108.5383.
    :param vmin: minimum WIMP velocity required for electron recoil energy of erec, float or np.ndarray
    :param t: J2000 timestamp, if not given a conservative velocity distribution is used, None or float
    :returns: mean inverse speed with units of time/distance attached
    """
    upper_lim = 1000 * nu.km/nu.s  # Integrand doesn't contribute beyond this point
    # upper_lim = np.inf
    
    # If scalar argument, convert to np array so logical indexing can be used
    if not isinstance(vmin, np.ndarray):
      scalar_arg = True
      _vmin = np.array([vmin], dtype=float)
    else:
      scalar_arg = False
      _vmin = vmin
    
    eta = np.zeros_like(_vmin, dtype=float)
    for idx in np.ndindex(_vmin.shape):
      if _vmin[idx] > upper_lim:
        continue
      integrand = lambda v: self.observed_speed_dist(v,t) / v
      eta[idx] = integrate.quad(integrand, a=_vmin[idx], b=upper_lim, limit=500, epsrel=1.49e-06, epsabs=0)[0]
    eta[eta<0] = 0 # Protect against negative values
    
    if scalar_arg:
      return eta[0]
    return eta
  
  
  # def analytic_mean_inv_speed(self, vmin, t=None):
  #   """DON'T USE!! Mean inverse speed, units of (time/distance)
  #      This is Eq. 11 in C. Savage, K. Freese, and P. Gondolo, Phys. Rev. D 74, 043531 (2006)
  #      This function is an alternative to mean_inv_speed(). While it is faster, there is a ~1% difference with mean_inv_speed() and breaks internal consistency with wimp_halo.py
  #   :param vmin: minimum WIMP velocity required for electron recoil energy of erec, scalar or np array
  #   :param t: J2000 timestamp, if not given a conservative velocity distribution is used
  #   :returns: mean inverse speed with units of time/distance attached
  #   """
  #   # If scalar argument, convert to np array so logical indexing can be used
  #   if not isinstance(vmin,np.ndarray):
  #     scalar_arg = True
  #     _vmin = np.array([vmin], dtype=float)
  #   else:
  #     scalar_arg = False
  #     _vmin = vmin
    
  #   v_0   = self.v_0        # Most probable speed of WIMPs relative to galaxy center (scalar)
  #   v_esc = self.v_esc      # Truncation speed for WIMP velocity distribution (scalar)
  #   v_obs = self.v_earth(t) # Motion of observer relative to rest frame of WIMP component (scalar)
    
  #   xx = _vmin / v_0
  #   yy = v_obs / v_0
  #   zz = v_esc / v_0
  #   N_esc = special.erf(zz) - 2*zz*np.exp(-zz**2)/np.pi**0.5 
    
  #   # Value of mean speed distribution
  #   eta = np.zeros_like(_vmin, dtype=float)
    
  #   idx1 = (xx<np.abs(yy-zz)) & (zz<yy)
  #   eta[idx1] = 1 / (v_0*yy)
    
  #   idx2 = (xx<np.abs(yy-zz)) & (zz>yy)
  #   eta[idx2] = ( special.erf(xx[idx2]+yy) - special.erf(xx[idx2]-yy) - 4*yy*np.exp(-zz**2)/np.pi**0.5 )  /  (2*N_esc*v_0*yy)
    
  #   idx3 = (xx>np.abs(yy-zz)) & (xx<yy+zz)
  #   eta[idx3] = ( special.erf(zz) - special.erf(xx[idx3]-yy) - 2*(yy+zz-xx[idx3])*np.exp(-zz**2)/np.pi**0.5 )  /  (2*N_esc*v_0*yy)
      
  #   idx4 = idx1 & idx2 & idx3
  #   eta[idx4] = 0
    
  #   if scalar_arg: # If vmin is a scalar, return a scalar
  #     return eta[0]
  #   return eta
  

if __name__ == '__main__':
  wr = WimpHalo()
  kms = nu.km/nu.s
  vs = np.linspace(0, 800 * kms, 100000)
  plt.plot(vs / kms, wr.galactic_v_dist(vs) * kms, label='Galactic frame', color='b')
  plt.plot(vs / kms, wr.observed_speed_dist(vs) * kms, label='Local frame', color='g')
  
  """
  mw = 0.5 * nu.GeV/nu.c0**2
  e_r_max = 2 * (mw * 39.948 * nu.amu / (mw + (39.948 * nu.amu)))**2 * (wr.v_max())**2 / (39.948 * nu.amu)
  e_r_min = 0.0001 * nu.eV
  ene = np.linspace(e_r_min, e_r_max, 100000)
  e_det = 0.1 * nu.keV
  vmin = np.zeros(0)
  v_dist = np.zeros(0)
  for e_r in ene:
    v = (39.948 * nu.amu * e_r + (mw * 39.948 * nu.amu / (mw + (39.948 * nu.amu))) * (e_det - 0.15 * e_r)) / ((mw * 39.948 * nu.amu / (mw + (39.948 * nu.amu))) * np.sqrt(2 * 39.948 * nu.amu * e_r))
    if v >= wr.v_max():
      v = wr.v_max()
    vmin = np.append(vmin, v / kms)
    v_dist = np.append(v_dist, wr.observed_speed_dist_xinran(v) * kms)
  #v_min = np.asarray(vmin)
  plt.plot(vmin, v_dist, label='Xinran', color='r')
  for v, vd in zip(vmin, v_dist):
    if v < 776:
      print(v,vd)
  """
  plt.show()
  print(wr.galactic_v_dist(220))
  

