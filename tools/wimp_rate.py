"""Elastic nuclear recoil rates
"""

import numpy as np
import numericalunits as nu
from scipy.interpolate import interp1d
from scipy.integrate import quad
import wimp_halo

class WimpNR:
  def __init__(self): 
    nu.reset_units('SI')    
    #Standard atomic weight of target
    self.an    = 39.948
    #Mass of nucleus (not nucleon!)
    self.mn    = self.an*nu.amu
    
    self.wr    = wimp_halo.WimpHalo()
    
  
  def reduced_mass(self, m1, m2):
    return m1 * m2 / (m1 + m2)


  def mu_nucleus(self, mw):
    """DM-nucleus reduced mass"""
    return self.reduced_mass(mw, self.mn)


  def e_max(self,mw, v, m_nucleus=None):
    """Kinematic nuclear recoil energy maximum
    :param mw: Wimp mass
    :param m_nucleus: Nucleus mass. Defaults to standard atomic mass.
    :param v: Wimp speed
    """
    if m_nucleus is None:
        m_nucleus = self.mn
    return 2 * self.reduced_mass(mw, m_nucleus)**2 * v**2 / m_nucleus


  def spherical_bessel_j1(self,x):
    """Spherical Bessel function j1 according to Wolfram Alpha"""
    return np.sin(x)/x**2 + - np.cos(x)/x


  #@wr.vectorize_first
  def helm_form_factor_squared(self, erec, anucl=None):
    """Return Helm form factor squared from Lewin & Smith

    Lifted from Andrew Brown's code with minor edits

    :param erec: nuclear recoil energy
    :param anucl: Nuclear mass number
    """
    if anucl is None:
        anucl = self.an
    
    en = erec / nu.keV
    
    if anucl <= 0:
        raise ValueError("Invalid value of A!")

    # TODO: Rewrite this so it doesn't use its internal unit system
    #  and hardcoded constants...

    # First we get rn squared, in fm
    mnucl = nu.amu/(nu.GeV/nu.c0**2)    # Mass of a nucleon, in GeV/c^2
    pi = np.pi
    c = 1.23*anucl**(1/3)-0.60
    a = 0.52
    s = 0.9
    rn_sq = c**2 + (7.0/3.0) * pi**2 * a**2 - 5 * s**2
    rn = np.sqrt(rn_sq)  # units fm
    mass_kev = anucl * mnucl * 1e6
    hbarc_kevfm = 197327  # hbar * c in keV *fm (from Wolfram alpha)

    # E in units keV, rn in units fm, hbarc_kev units keV.fm
    # Formula is spherical bessel fn of Q=sqrt(E*2*Mn_keV)*rn
    q = np.sqrt(en*2.*mass_kev)
    qrn_over_hbarc = q*rn/hbarc_kevfm
    sph_bess = self.spherical_bessel_j1(qrn_over_hbarc)
    retval = 9. * sph_bess * sph_bess / (qrn_over_hbarc*qrn_over_hbarc)
    qs_over_hbarc = q*s/hbarc_kevfm
    retval *= np.exp(-qs_over_hbarc*qs_over_hbarc)
    return retval

  def mediator_factor(self, erec, m_med):
    if m_med == float('inf'): return 1
    q = (2 * self.mn * erec)**0.5
    return m_med**4 / (m_med**2 + (q/nu.c0)**2)**2

  def sigma_erec(self, erec, v, mw, sigma_nucleon, interaction='SI', m_med=float('inf')):
    """Differential elastic WIMP-nucleus cross section
    (dependent on recoil energy and wimp-earth speed v)

    :param erec: recoil energy
    :param v: WIMP speed (earth/detector frame)
    :param mw: Mass of WIMP
    :param sigma_nucleon: WIMP-nucleon cross-section
    :param interaction: string describing DM-nucleus interaction.
    See rate_wimps for options.
    :param m_med: Mediator mass. If not given, assumed much heavier than mw.
    """
    sigma_nucleus = (sigma_nucleon
                     * (self.mu_nucleus(mw) / self.reduced_mass(nu.amu, mw))**2
                     * self.an**2)
    result = (sigma_nucleus
              / self.e_max(mw, v)
              * self.helm_form_factor_squared(erec, anucl=self.an))


    return result * self.mediator_factor(erec, m_med)




  def vmin_elastic(self, erec, mw):
    """Minimum WIMP velocity that can produce a recoil of energy erec
    :param erec: recoil energy
    :param mw: Wimp mass
    """
    return np.sqrt(self.mn * erec / (2 * self.mu_nucleus(mw)**2))


  #@wr.vectorize_first
  
  def rate_elastic(self, erec, mw, sigma_nucleon,interaction='SI', m_med=float('inf'), t=None, **kwargs):
  #def rate_elastic(self, erec, mw, sigma_nucleon, m_med=float('inf'), t=None):
    """Differential rate per unit detector mass and recoil energy of
    elastic WIMP scattering

    :param erec: recoil energy
    :param mw: WIMP mass
    :param sigma_nucleon: WIMP/nucleon cross-section
    :param interaction: string describing DM-nucleus interaction,
    see sigma_erec for options
    :param m_med: Mediator mass. If not given, assumed very heavy.
    :param t: A J2000.0 timestamp.
    If not given, conservative velocity distribution is used.
    :param progress_bar: if True, show a progress bar during evaluation
    (if erec is an array)

    Further kwargs are passed to scipy.integrate.quad numeric integrator
    (e.g. error tolerance).

    Analytic expressions are known for this rate, but they are not used here.
    """
    
    #  avoid seg fault for erec=0:
    #erec = np.where(erec>0,erec,1e-4)
    
    #v_min = self.vmin_elastic(erec, mw)
    v_esc = self.wr.v_max(t)
    
    #v_min = np.where(v_min >= self.wr.v_max(t), 0, v_min)
    
    #if v_min >= self.wr.v_max(t):
    #    return 0
    
    
    rel = np.zeros(0)
    for e in erec:
      if e == 0: 
        rel = np.append(rel,0)
        continue
      
      v_min = self.vmin_elastic(e, mw)
      if v_min >= v_esc:
        rel = np.append(rel,0)
        continue
        
      integrand = lambda v: self.sigma_erec(e, v, mw, sigma_nucleon,  m_med) * v * self.wr.observed_speed_dist(v, t)
      rel = np.append(rel, self.wr.rho_dm / mw * (1 / self.mn) * quad(integrand, v_min, v_esc,**kwargs)[0])
     
    
    #def integrand(v):
    #    return (self.sigma_erec(erec, v, mw, sigma_nucleon,  m_med) * v * self.wr.observed_speed_dist(v, t))

    #print([integrand(5000,x) for x in erec])
    #print( quad(integrand, 5000, self.wr.v_max(t),**kwargs)[0])
    
    #rate = np.array([self.wr.rho_dm / mw * (1 / self.mn) * quad(integrand, x, v_esc,**kwargs)[0] for x in v_min])
    #rate = np.where(v_min < v_esc, rate, 0)
    
    #return self.wr.rho_dm / mw * (1 / self.mn) * quad(integrand, v_min, self.wr.v_max(t),**kwargs)[0]
    return rel

  def rate_wimp_nr(self, erec, mw, sigma_nucleon, m_med=float('inf'), t=None, **kwargs):
      """Differential rate per (ton year keV) of WIMP-nucleus scattering.
      :param es: Recoil energies in keV
      :param mw: WIMP mass in GeV/c^2
      :param sigma_nucleon: WIMP-nucleon cross-section in cm^2
      :param m_med: Medator mass in GeV/c^2. If not given, assumed very heavy.
      :param t: A J2000.0 timestamp. If not given,
      conservative velocity distribution is used.
      :returns: numpy array of same length as es
      Further arguments are as for rate_wimp; see docstring of rate_wimp.
      """
      return (self.rate_elastic(erec=erec * nu.keV,
                        mw=mw * nu.GeV/nu.c0**2,
                        sigma_nucleon=sigma_nucleon * nu.cm**2,
                        m_med=m_med * nu.GeV/nu.c0**2, t=t, **kwargs)
              * (nu.keV * (1000 * nu.kg) * nu.year))

if __name__ == '__main__':
  
  import matplotlib.pyplot as plt
  wimprates = WimpNR()
  
  Mw        = 10           # GeV/c2
  CS        = 3e-41        # cm^2
  maxene    = 20           # keV
  step      = 0.01         # keV
  livetime  = 432          # days
  mass      = 0.41883*46.7 # kg


  perkevtonyear = ((1000 * nu.kg)**-1 * nu.year**-1 * nu.keV**-1)
  
  livetime  = livetime*nu.day/nu.year    # transform in years
  mass      = mass*nu.kg/(1000 * nu.kg)  # transform in tons
  
  # calculate the exposure in ton year
  exposure_kg_day = round(mass*livetime*nu.year/nu.day*1000,1)
  
  # create an array of NR energies in keV
  ene = np.linspace(0,maxene,int(maxene/step)+1)
  
  # evaluate CS for each NR energy 
  #cs =  [wimprates.rate_wimp_nr(erec=x,  mw = Mw, sigma_nucleon = CS)*mass*livetime*step for x in ene] 
  cs =  wimprates.rate_wimp_nr(erec=ene,  mw = Mw, sigma_nucleon = CS)*mass*livetime*step
  
  # print the expected number of events in the given exposure
  print("WIMP mass:",Mw, "GeV/c2; CS:", CS," cm2")
  print("number of events in", exposure_kg_day, "kg day:", round(np.sum(cs),0))

  # plot the CS
  plt.xlabel('[keV]')
  plt.ylabel("ev / %d kg day keV"%exposure_kg_day)
  plt.title('Mw = %.1f GeV/c^2; sigma = 1e%.2f cm^2'%(Mw, round(np.log10(CS),2)))
  plt.plot(ene,cs)
  plt.show()

