# Implementation of the Lewin-Smith calculations in Astrop. Phys. 6 (1996) 87-112
# Valerio Ippolito - INFN Sezione di Roma
import numpy as np
import numericalunits as nu
from scipy import integrate
from scipy.special import erf, spherical_jn

dru = 1.0/(nu.keV * nu.kg * nu.day)
tru = 1.0/(nu.kg * nu.day)
iru = 1.0/(nu.kg * nu.day)
barn = 1e-24 * np.power(nu.cm, 2)
hbar = 197.327 * nu.MeV * nu.fm / nu.c0

def dR_dEr_simple(Er, E0, Md, Mt, R0):
  '''
      Eq. 1.1
      Differential rate for a detector stationary in the Galaxy

      Er: nuclear recoil energy
      E0: most probable incident kinetic energy
      Md: dark matter mass
      Mt: target nucleus mass
      R: event rate per unit mass
      R0: total event rate
  '''
  r = 4*Md*Mt/np.power(Md+Mt,2)
  E0r = E0 * r
  return R0 / (E0r) * np.exp(-Er / (E0r))

def dR_dEr(E, R0, S, F, I):
  '''
      Eq. 1.2
      Differential rate, general case

      E: observed energy
      R0: total event rate (unmodified rate for a stationary detector)
      S: modified spectral function accounting for
          - detector movement w.r.t. Galaxy
          - difference between true and observed recoil energy (NR vs ER)
          - target composition
          - instrumental resolution and threshold effects
      F: form factor correction
      I: interaction function for spin-dependent / spin-independent interactions
  '''
  return R0 * S(E) * np.power(F(E), 2) * I

def velocity_maxwell(v, v_earth, v0):
  return np.exp(- np.power(v+v_earth, 2) / np.power(v0, 2))

def k0_maxwell(v0):
  return np.power(np.pi * np.power(v0, 2), 3./2.)

def k1_maxwell(v0, v_esc):
  return k0_maxwell(v0) * (erf(v_esc/v0) - 2/np.power(np.pi, 0.5) * v_esc/v0 * np.exp(-np.power(v_esc/v0, 2)))

def dR_simple(A, sigma, v, dn):
  ''' 
  dNint = sigma phi Nb = sigma (dn v) * (rho NA/A * V) = sigma dn v M NA/A
  and so
  dR = dNint/M = sigma dn v NA/A

  A: atomic mass
  sigma: cross-section
  v: velocity
  dn: input particle density
  '''
  N_avogadro = 6.02e26 / nu.kg
  return N_avogadro / (A/nu.amu) * sigma * v * dn

def dR_simple2(A, sigma0, n0, avg_velocity):
  return dR_simple(A=A, sigma=sigma0, n=n0, velocity=avg_velocity)

def R0(A, rhoD, Md, sigma0, v0):
  ''' Eq. 3.1 '''
  N_avogadro = 6.02e26 / nu.kg
  return 2/np.power(np.pi, 0.5) * N_avogadro / (A/nu.amu) * rhoD/Md * sigma0 * v0

def R0_approx(Md, Mt, sigma0, rhoD, v0):
  ''' Eq. 3.7 '''
  return 503 * tru / ((Md/(nu.GeV/np.power(nu.c0,2))) * (Mt/(nu.GeV/np.power(nu.c0,2)))) * (sigma0 / (1e-12*barn)) * (rhoD / (0.4 * nu.GeV / np.power(nu.c0, 2) / np.power(nu.cm, 3))) * (v0 / (230 * nu.km / nu.s))

def R_0_vesc(v_esc, A, rhoD, Md, sigma0, v0):
  ''' Eq. 3.3 '''
  argument = np.power(v_esc/v0, 2)
  return R0(A, rhoD, Md, sigma0, v0) * k0_maxwell(v0)/k1_maxwell(v0, v_esc) * (1- (1 + argument) * np.exp(-argument))

def R_vE_inf(vE, A, rhoD, Md, sigma0, v0):
  ''' Eq. 3.4 '''
  argument = np.power(vE/v0, 2)
  return 0.5 * R0(A, rhoD, Md, sigma0, v0) * (np.power(np.pi, 0.5) * (vE/v0 + 0.5 * v0/vE) * np.erf(vE/v0) + np.exp(-argument))

def R_vE_vesc(vE, v_esc, A, rhoD, Md, sigma0, v0):
  ''' Eq. 3.5 '''
  argument = np.power(v_esc/v0, 2)
  Rzero = R0(A, rhoD, Md, sigma0, v0)
  return 0.5 * Rzero * k0_maxwell(v0)/k1_maxwell(v0, v_esc) * (R_vE_inf(vE=vE, A=A, rhoD=rhoD, Md=Md, sigma0=sigma0, v0=v0)/Rzero - (argument + 1/3. * np.power(vE/v0, 2) + 1) * np.exp(-argument))

def vE_approx(time_from_mar2):
  ''' Eq. 3.6 '''
  return (244 + 15 * np.sin(2 * np.pi * time_from_mar2 / nu.year)) * nu.km / nu.s

def kinetic_energy(Md, v):
  return 0.5 * Md * np.power(v, 2)

def r_factor(Md, Mt):
  ''' Eq. 3.8 '''
  return 4 * Md * Mt / np.power(Md + Mt, 2)

def recoil_energy(Md, Mt, v, theta):
  E = kinetic_energy(Md, v)
  r = r_factor(Md, Mt)
  return 0.5 * E * r * (1 - np.cos(theta))

def min_incident_energy(Md, Mt, Er):
  r = r_factor(Md, Mt)
  return Er / r

def E0(Md, v0):
  return 0.5 * Md * np.power(v0, 2)

def v_min(Md, Mt, Er, v0):
  Ezero = E0(Md, v0)
  r = r_factor(Md, Mt)
  return np.sqrt(Er / (Ezero * r)) * v0

def rate_integral(A, rhoD, Md, Mt, sigma0, v0, vE, v_esc):
  ''' Eq. 3.9 '''
  Rzero = R0(A, rhoD, Md, sigma0, v0)
  Ezero = E0(Md, v0)
  r = r_factor(Md, Mt)
  kzero = k0_maxwell(v0)
  kone = k1_maxwell(v0, v_esc)

  constant = Rzero / (Ezero * r) * kzero / kone * 1 / (2 * np.pi * np.power(v0, 2))
  integrand = lambda x: 1/x * velocity_maxwell(v, vE) # TODO: should be vectorial

  vmin = v_min(Md, Mt, Er, v0)
  vmax = v_esc

  return constant * integrate.quad(integrand, vmin, vmax)[0]
  
def rate_0_infty(A, rhoD, Md, Mt, sigma0, v0, Er):
  ''' Eq. 3.10 '''
  Rzero = R0(A, rhoD, Md, sigma0, v0)
  Ezero = E0(Md, v0)
  r = r_factor(Md, Mt)
  Ezero_r = Ezero * r

  return Rzero / (Ezero_r) * np.exp(-Er/(Ezero_r))

def rate_0_vesc(A, rhoD, Md, Mt, sigma0, v0, Er, v_esc):
  ''' Eq. 3.11 '''
  kzero = k0_maxwell(v0)
  kone = k1_maxwell(v0, v_esc)
  Rzero = R0(A, rhoD, Md, sigma0, v0)
  Ezero = E0(Md, v0)
  r = r_factor(Md, Mt)
  Ezero_r = Ezero * r

  return kzero / kone * (rate_0_infty(A, rhoD, Md, Mt, sigma0, v0, Er) - Rzero/(Ezero_r) * np.exp(-np.power(v_esc/v0, 2)))

def rate_vE_infty(A, rhoD, Md, Mt, sigma0, v0, Er, vE):
  ''' Eq. 3.12 '''
  Rzero = R0(A, rhoD, Md, sigma0, v0)
  Ezero = E0(Md, v0)
  r = r_factor(Md, Mt)
  Ezero_r = Ezero * r
  vmin = v_min(Md, Mt, Er, v0)

  return Rzero / (Ezero_r) * np.sqrt(np.pi)/4 * v0/vE * (erf((vmin + vE)/v0) - erf((vmin - vE)/v0))

def rate_vE_infty_approx(A, rhoD, Md, Mt, sigma0, v0, Er, c1, c2):
  ''' Eq. 3.14 '''
  Rzero = R0(A, rhoD, Md, sigma0, v0)
  Ezero = E0(Md, v0)
  r = r_factor(Md, Mt)
  Ezero_r = Ezero * r

  return c1 * Rzero / (Ezero_r) * np.exp(-c2 * Er / Ezero_r)

def integrated_rate_vE_infty_approx(A, rhoD, Md, Mt, sigma0, v0, Er_min, Er_max, c1, c2):
  ''' Eq. 3.15 '''
  Rzero = R0(A, rhoD, Md, sigma0, v0)
  Ezero = E0(Md, v0)
  r = r_factor(Md, Mt)
  Ezero_r = Ezero * r

  return Rzero * c1/c2 * (np.exp(-c2 * Er_min / Ezero_r) - np.exp(-c2 * Er_max / Ezero_r))

def rate_dEr_dcospsi_vE_infty(A, rhoD, Md, Mt, sigma0, v0, Er, vE, cospsi):
  ''' Eq. 3.16 '''
  Rzero = R0(A, rhoD, Md, sigma0, v0)
  Ezero = E0(Md, v0)
  r = r_factor(Md, Mt)
  Ezero_r = Ezero * r
  vmin = v_min(Md, Mt, Er, v0)

  return 0.5 * Rzero / Ezero_r * np.exp(-np.power((vE * cospsi - vmin)/v0, 2))

def rate_fwd_vE_infty(A, rhoD, Md, Mt, sigma0, v0, Er, vE):
  Rzero = R0(A, rhoD, Md, sigma0, v0)
  Ezero = E0(Md, v0)
  r = r_factor(Md, Mt)
  Ezero_r = Ezero * r
  vmin = v_min(Md, Mt, Er, v0)

  return Rzero / Ezero_r * np.sqrt(np.pi)/4 * v0/vE * (erf(vmin/v0) - erf((vmin - vE)/v0))

def rate_bkw_vE_infty(A, rhoD, Md, Mt, sigma0, v0, Er, vE):
  Rzero = R0(A, rhoD, Md, sigma0, v0)
  Ezero = E0(Md, v0)
  r = r_factor(Md, Mt)
  Ezero_r = Ezero * r
  vmin = v_min(Md, Mt, Er, v0)

  return Rzero / Ezero_r * np.sqrt(np.pi)/4 * v0/vE * (erf((vmin + vE)/v0) - erf(vmin/v0))

def integrated_rate_vE_infty(A, rhoD, Md, Mt, sigma0, v0, Er_min, Er_max, vE):
  ''' Energy integral of Eq. 3.12 '''
  integrand = lambda x: rate_vE_infty(A=A, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er=x, vE=vE)

  return integrate.quad(integrand, Er_min, Er_max)[0]

def integrated_rate_fwd_vE_infty(A, rhoD, Md, Mt, sigma0, v0, Er_min, Er_max, vE):
  integrand = lambda x: rate_fwd_vE_infty(A=A, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er=x, vE=vE)

  return integrate.quad(integrand, Er_min, Er_max)[0]

def integrated_rate_bkw_vE_infty(A, rhoD, Md, Mt, sigma0, v0, Er_min, Er_max, vE):
  integrand = lambda x: rate_bkw_vE_infty(A=A, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er=x, vE=vE)

  return integrate.quad(integrand, Er_min, Er_max)[0]

def rate_par_vE_infty(A, rhoD, Md, Mt, sigma0, v0, Er, vE):
  ''' cos(psi) integral of Eq. 3.16 parallel to the target trajectory '''
  integrand = lambda x: np.abs(x) * rate_dEr_dcospsi_vE_infty(A=A, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er=Er, vE=vE, cospsi=x)

  return integrate.quad(integrand, -1, 1)[0]

def rate_perp_vE_infty(A, rhoD, Md, Mt, sigma0, v0, Er, vE):
  ''' cos(psi) integral of Eq. 3.16 perpendicular to the target trajectory '''
  integrand = lambda x: np.sqrt(1 - np.power(x, 2)) * rate_dEr_dcospsi_vE_infty(A=A, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er=Er, vE=vE, cospsi=x)

  return integrate.quad(integrand, -1, 1)[0]

def rate_dpsi_vE_infty(A, rhoD, Md, Mt, sigma0, v0, Er_min, Er_max, vE, cospsi):
  ''' R0 times Eq. 3.17, i.e. Er integral of 3.16 '''
  Rzero = R0(A, rhoD, Md, sigma0, v0)
  Ezero = E0(Md, v0)
  r = r_factor(Md, Mt)
  Ezero_r = Ezero * r
  v1 = np.sqrt(Er_min / Ezero_r) * v0
  v2 = np.sqrt(Er_max / Ezero_r) * v0

  return Rzero * (0.5 * (np.exp(-np.power((v1 - vE * cospsi)/v0, 2)) - np.exp(-np.power((v2 - vE * cospsi)/v0, 2))) + np.sqrt(np.pi)/2 * vE/v0 * cospsi * (erf((v2 - vE * cospsi)/v0) - erf((v1 - vE * cospsi)/v0)))

def integrated_rate_dcospsi_par_vE_infty(A, rhoD, Md, Mt, sigma0, v0, Er_min, Er_max, vE):
  ''' cos(psi) integral of R0 times Eq. 3.17 parallel to the target trajectory '''
  integrand = lambda x: np.abs(x) * rate_dpsi_vE_infty(A=A, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er_min=Er_min, Er_max=Er_max, vE=vE, cospsi=x)

  return integrate.quad(integrand, -1, 1)[0]

def integrated_rate_dcospsi_perp_vE_infty(A, rhoD, Md, Mt, sigma0, v0, Er_min, Er_max, vE):
  ''' cos(psi) integral of Eq. 3.17 perpendicular to the target trajectory '''
  integrand = lambda x: np.sqrt(1 - np.power(x, 2)) * rate_dpsi_vE_infty(A=A, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er_min=Er_min, Er_max=Er_max, vE=vE, cospsi=x)

  return integrate.quad(integrand, -1, 1)[0]

def integrated_rate_par_vE_infty(A, rhoD, Md, Mt, sigma0, v0, Er_min, Er_max, vE):
  ''' brute-force double (cos(psi) and Er) integral of Eq. 3.16 parallel to the target trajectory '''
  integrand = lambda y, x: np.abs(x) * rate_dEr_dcospsi_vE_infty(A=A, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er=y, vE=vE, cospsi=x)

  return integrate.dblquad(integrand, -1, 1, Er_min, Er_max)[0]

def integrated_rate_perp_vE_infty(A, rhoD, Md, Mt, sigma0, v0, Er_min, Er_max, vE):
  ''' brute-force double (cos(psi) and Er) integral of Eq. 3.16 perpendicular to the target trajectory '''
  integrand = lambda y, x: np.sqrt(1 - np.power(x, 2)) * rate_dEr_dcospsi_vE_infty(A=A, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er=y, vE=vE, cospsi=x)

  return integrate.dblquad(integrand, -1, 1, Er_min, Er_max)[0]

def rate_vE_vesc(A, rhoD, Md, Mt, sigma0, v0, Er, vE, v_esc):
  ''' Eq. 3.13 '''
  kzero = k0_maxwell(v0)
  kone = k1_maxwell(v0, v_esc)
  Rzero = R0(A, rhoD, Md, sigma0, v0)
  Ezero = E0(Md, v0)
  r = r_factor(Md, Mt)
  Ezero_r = Ezero * r

  return kzero / kone * (rate_vE_infty(A, rhoD, Md, Mt, sigma0, v0, Er, vE) - Rzero/(Ezero_r) * np.exp(-np.power(v_esc/v0, 2)))

def nuclear_mass(A):
  #return 0.932 * A * nu.GeV/np.power(nu.c0, 2)
  return A * nu.amu

def momentum_transfer(Mt, Er):
  return np.sqrt(2 * Mt * Er)

def qrn(A, a_n, b_n, Er):
  ''' Dimensionful version of Eq. 4.1
      NB: most formulas use the dimensionless quantity qrn/hbar
  '''
  rn = r_n_approx(A=A, a_n=a_n, b_n=b_n)
  q = momentum_transfer(Mt=nuclear_mass(A), Er=Er)
  return q * rn

def form_factor_integral(q, rho):
  ''' Generic form factor expressed as integral of the density distribution
      of the scattering centers, in the first Born (plane wave) approximation
  '''
  integrand = lambda x: x * np.sin(q*x) * rho(x)
  return 4 * np.pi / q * integrate.quad(integrand, 0, np.infty)[0]

def thin_shell_form_factor(qrn):
  ''' Eq. 4.2, i.e. a first example of SD form factor  '''
  qrn = qrn/hbar
  return spherical_jn(n=0, z=qrn, derivative=False)
 #return np.sin(qrn)/qrn

def solid_sphere_form_factor(qrn):
  ''' Eq. 4.3, i.e. a first example of SI form factor  '''
  qrn = qrn/hbar
  return 3 * spherical_jn(n=1, z=qrn, derivative=False) / qrn
 #return 3*(np.sin(qrn) - qrn * np.cos(qrn))/np.power(qrn, 3)

def approx_form_factor_squared(qrn, alpha):
  ''' Eq. 4.4 
    - alpha = 1/3: exact form factor for a gaussian scatterer
      with r_rms = r_n; approximates Eq. 4.2 for small qrn
    - alpha = 1/5: approximates Eq. 4.3 for qrn below 3-4
  '''
  qrn = qrn/hbar
  return np.exp(-alpha * np.power(qrn, 2))

def approx_SD_form_factor_squared_q(q, A):
  ''' Eq. 4.5 '''
  rn = 1.0 * np.power(A, 1/3.) * nu.fm
  qrn = q * rn / hbar
  return approx_SD_form_factor_squared(qrn)

def approx_SD_form_factor_squared(qrn, A):
  ''' The qrn-dependent part of Eq. 4.5 '''
  qrn = qrn/hbar
  return np.where((qrn < 2.55) | (qrn > 4.5), np.power(spherical_jn(n=0, z=qrn, derivative=False), 2), 0.047)

def fermi_distribution(r, rho0, c, a):
  ''' Eq. 4.6 '''
  return rho0 / (1 + np.exp((r - c) / a))
 
def helm_form_factor(q, A, a, s):
  ''' Eq. 4.7 '''
  rn = np.sqrt(r_n_squared(A, a, s))
  qrn = q * rn / hbar
  qs = q * s / hbar
  return 3 * spherical_jn(n=1, z=qrn, derivative=False)/qrn * np.exp(-0.5 * np.power(qs, 2))

def helm_form_factor_simpler(q, A, s, a_n, b_n):
  ''' Eq. 4.7 with approximated r_n '''
  rn = r_n_approx(A=A, a_n=a_n, b_n=b_n)
  qrn = q * rn / hbar
  qs = q * s / hbar
  return 3 * spherical_jn(n=1, z=qrn, derivative=False)/qrn * np.exp(-0.5 * np.power(qs, 2))

def uniform_sphere_r_var(rn):
  ''' (r_rms)^2 for an uniform sphere '''
  return 3/5. * np.power(rn, 2)

def fermi_r_var(c, a):
  ''' Eq. 4.8, i.e. (r_rms)^2 for a Fermi distribution of scattering centers '''
  return 3/5. * np.power(c, 2) + 7/5. * np.power(np.pi * a, 2)

def helm_r_var(rn, s):
  ''' Eq. 4.9, i.e. (r_rms)^2 for the Helm form factor '''
  return 3/5. * np.power(rn, 2) + 3 * np.power(s, 2)

def c_param(A):
  ''' Eq. 4.10 '''
  return (1.23 * np.power(A, 1/3.) - 0.60) * nu.fm

def r_n_squared(A, a, s):
  ''' Eq. 4.11, i.e. r_n for Eq. 4.7 (helm_form_factor)  '''
  c = c_param(A)
  return np.power(c, 2) + 7/3. * np.power(np.pi * a, 2) - 5 * np.power(s, 2)

def r_n_approx(A, a_n, b_n):
  ''' Approximation of 4.11 for most A is a_n=1.14 fm, b_n=0 fm'''
  return a_n * np.power(A, 1/3.) + b_n

def reduced_mass(mA, mB):
  return (mA*mB)/(mA+mB)

def spin_independent_factor(A, Md):
  redmass_factor = np.power(reduced_mass(mA=A, mB=Md) / reduced_mass(mA=nu.amu, mB=Md), 2)
  return np.power(A/nu.amu, 2) * redmass_factor


if __name__ == '__main__':
  import matplotlib.pyplot as plt
  AAr_nounits = 40
  AAr = AAr_nounits * nu.amu
  v0 = 230 * nu.km/nu.s
  v_esc = 600 * nu.km/nu.s
  rhoD = 0.4 * nu.GeV / np.power(nu.c0, 2) / np.power(nu.cm, 3)
  Md = 100 * nu.GeV / np.power(nu.c0, 2)
  Mt = nuclear_mass(AAr_nounits)
  sigma0 = 1e-12 * barn
  s = 0.9 * nu.fm
  a = 0.52 * nu.fm
  a_n = 1.14 * nu.fm # i.e. we get back r_n_approx for Eq. 4.1
  b_n = 0 * nu.fm # i.e. we get back r_n_approx for Eq. 4.1

  from datetime import date
  mar2 = date(2022, 3, 2)
  d1 = date(2022, 8, 30)
  dt = d1 - mar2
  dt_years = dt.days*nu.day

  print(f'k1/k0 = {k1_maxwell(v0=v0, v_esc=v_esc) / k0_maxwell(v0=v0)}')
  print(f'R_0_vesc/R0 = {R_0_vesc(v0=v0, v_esc=v_esc, A=AAr, rhoD=rhoD, Md=Md, sigma0=sigma0)/R0(A=AAr, rhoD=rhoD, Md=Md, sigma0=sigma0, v0=v0)}')
  print(f'Aug 30, 2022 is {dt.days} days ({dt_years} years) from March 2: vE ~ {vE_approx(time_from_mar2=dt_years)/nu.km*nu.s} km/s')
  print(f'R0 = {R0(A=AAr, rhoD=rhoD, Md=Md, sigma0=sigma0, v0=v0)/tru} tru, R0_approx = {R0_approx(Md=Md, Mt=Mt, sigma0=sigma0, rhoD=rhoD, v0=v0)/tru} tru')

  Er = np.linspace(0, 2000, 10000) * nu.keV
  r = r_factor(Md=Md, Mt=Mt)
  Ezero = E0(Md=Md, v0=v0)
  Ezero_r = Ezero * r
  E_over_E0_r = Er / Ezero_r
  Rzero = R0(A=AAr, rhoD=rhoD, Md=Md, sigma0=sigma0, v0=v0)

  dt_jun = (date(2022, 6, 1) - mar2).days * nu.day
  dt_jun_avg = np.linspace(dt_jun, dt_jun + 30 * nu.day/nu.year, 30)
  vE_jun_avg = vE_approx(time_from_mar2=dt_jun_avg)
  rate_jun_avg = np.array([rate_vE_vesc(A=AAr, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er=Er, vE=vE, v_esc=v_esc) for vE in vE_jun_avg]).mean(axis=0)

  dt_dec = (date(2022, 12, 1) - mar2).days * nu.day
  dt_dec_avg = np.linspace(dt_dec, dt_dec + 31 * nu.day/nu.year, 31)
  vE_dec_avg = vE_approx(time_from_mar2=dt_dec_avg)
  rate_dec_avg = np.array([rate_vE_vesc(A=AAr, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er=Er, vE=vE, v_esc=v_esc) for vE in vE_dec_avg]).mean(axis=0)

  dt_year_avg = np.linspace(0, 365, 365) * nu.day
  vE_year_avg = vE_approx(time_from_mar2=dt_year_avg)
  rate_year_avg = np.array([rate_vE_vesc(A=AAr, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er=Er, vE=vE, v_esc=v_esc) for vE in vE_year_avg]).mean(axis=0)

  print([vE_approx(x)/nu.km*nu.s for x in [0, 0.5, 1, 1.5, 2]])
  plt.plot(dt_year_avg/nu.year, vE_year_avg/nu.km*nu.s)
  plt.xlabel('years from Mar 2')
  plt.ylabel('earth velocity [km/s]')
  plt.title(f'Cross-check of Earth velocity vs time for A={AAr_nounits}')
  plt.show()

  plt.plot(E_over_E0_r, rate_jun_avg * (Ezero_r / Rzero), label='Jun average')
  plt.plot(E_over_E0_r, rate_dec_avg * (Ezero_r / Rzero), label='Dec average')
  plt.plot(E_over_E0_r, rate_year_avg * (Ezero_r / Rzero), label='year average')
  plt.legend()
  plt.xlabel('$E_{R}/(E_0 r)$')
  plt.ylabel('$(E_0 r/R_0) dR/dE_{R}$')
  plt.xlim(0, 10)
  plt.ylim(0, 1)
  plt.title('Fig. 2 for A={AAr_nounits}')
  plt.show()


  plt.plot(E_over_E0_r, (rate_jun_avg - rate_year_avg) * (Ezero_r / Rzero), label='Jun average')
  plt.plot(E_over_E0_r, (rate_dec_avg - rate_year_avg) * (Ezero_r / Rzero), label='Dec average')
  plt.plot(E_over_E0_r, (rate_year_avg - rate_year_avg) * (Ezero_r / Rzero), label='year average')
  plt.legend()
  plt.xlabel('$E_{R}/(E_0 r)$')
  plt.ylabel('$(E_0 r/R_0) dR/dE_{R}$ minus year average')
  plt.xlim(0.74, 0.84)
# plt.ylim(-1e-3, 5e-4)
  plt.title('Fig. 2 inset for A={AAr_nounits}')
  plt.show()

  rate_vE_infty_ = np.array([rate_vE_infty(A=AAr, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er=Er, vE=vE) for vE in vE_year_avg]).mean(axis=0)
  rate_vE_infty_approx_ = rate_vE_infty_approx(A=AAr, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er=Er, c1=0.751, c2=0.561)
  plt.plot(E_over_E0_r, rate_vE_infty_ * (Ezero_r / Rzero), label='Eq. 3.12 (year average)')
  plt.plot(E_over_E0_r, rate_year_avg * (Ezero_r / Rzero), label='Eq. 3.14 ($c_1=0.751, c_2=0.561$)')
  plt.legend()
  plt.xlabel('$E_{R}/(E_0 r)$')
  plt.ylabel('$(E_0 r/R_0) dR(v_E,\infty)/dE_{R}$')
  plt.xlim(0, 10)
  plt.ylim(0, 1)
  plt.title('Cross-check of 3.14 with fixed')
  plt.show()

  exposure = 16660 * nu.kg * nu.day
  print(f'Integrated rate between 80 and 200 keV for A={AAr_nounits} (vE year average, v_esc=infty, approximated): {integrated_rate_vE_infty_approx(A=AAr, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er_min=80*nu.keV, Er_max=200*nu.keV, c1=0.751, c2=0.561)*exposure} for an exposure of {exposure/nu.tonne/nu.year} t yr')

  print('\n')
  print(f'Table 1 for A={AAr_nounits}')
  print('Er/(E0 r), R/R0 (tot, Jun), R/R0 (tot, Dec), R/R0 (fwd, Jun), R/R0 (bkw, Jun), R/R0 (fwd, Dec), R/R0 (bkw, Dec)')

  Er_over_E0_r = [ [0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.5], [0.5, 0.7], [0.7, 1.0], [1, 2], [2, 3], [3, 5], [5, 7], [7, 10], [0, 10] ]

  for (Er_over_E0_r_min, Er_over_E0_r_max) in Er_over_E0_r:
    Er_min, Er_max = Er_over_E0_r_min * Ezero_r, Er_over_E0_r_max * Ezero_r

    rate_jun_tot = np.mean([integrated_rate_vE_infty(A=AAr, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er_min=Er_min, Er_max=Er_max, vE=vE) for vE in vE_jun_avg])
    rate_dec_tot = np.mean([integrated_rate_vE_infty(A=AAr, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er_min=Er_min, Er_max=Er_max, vE=vE) for vE in vE_dec_avg])

    rate_jun_fwd = np.mean([integrated_rate_fwd_vE_infty(A=AAr, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er_min=Er_min, Er_max=Er_max, vE=vE) for vE in vE_jun_avg])
    rate_dec_fwd = np.mean([integrated_rate_fwd_vE_infty(A=AAr, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er_min=Er_min, Er_max=Er_max, vE=vE) for vE in vE_dec_avg])

    rate_jun_bkw = np.mean([integrated_rate_bkw_vE_infty(A=AAr, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er_min=Er_min, Er_max=Er_max, vE=vE) for vE in vE_jun_avg])
    rate_dec_bkw = np.mean([integrated_rate_bkw_vE_infty(A=AAr, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er_min=Er_min, Er_max=Er_max, vE=vE) for vE in vE_dec_avg])

    print(f'{Er_over_E0_r_min:.1f}-{Er_over_E0_r_max:.1f}, {rate_jun_tot/Rzero:.3f}, {rate_dec_tot/Rzero:.3f}, {rate_jun_fwd/Rzero:.3f}, {rate_jun_bkw/Rzero:.3f}, {rate_dec_fwd/Rzero:.3f}, {rate_dec_bkw/Rzero:.3f}')


  print('\n')
  print('Table 2')
  print('Er/(E0 r), R/R0 (tot, Jun), R/R0 (tot, Dec), R/R0 (fwd, Jun), R/R0 (bkw, Jun), R/R0 (fwd, Dec), R/R0 (bkw, Dec)')
  use_brute_force_2D_integrals = False
  for (Er_over_E0_r_min, Er_over_E0_r_max) in Er_over_E0_r:
    continue # TODO remove
    Er_min, Er_max = Er_over_E0_r_min * Ezero_r, Er_over_E0_r_max * Ezero_r

    if use_brute_force_2D_integrals:
      rate_jun_par = np.mean([integrated_rate_par_vE_infty(A=AAr, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er_min=Er_min, Er_max=Er_max, vE=vE) for vE in vE_jun_avg])
      rate_jun_perp = np.mean([integrated_rate_perp_vE_infty(A=AAr, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er_min=Er_min, Er_max=Er_max, vE=vE) for vE in vE_jun_avg])

      rate_dec_par = np.mean([integrated_rate_par_vE_infty(A=AAr, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er_min=Er_min, Er_max=Er_max, vE=vE) for vE in vE_dec_avg])
      rate_dec_perp = np.mean([integrated_rate_perp_vE_infty(A=AAr, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er_min=Er_min, Er_max=Er_max, vE=vE) for vE in vE_dec_avg])

      rate_year_par = np.mean([integrated_rate_par_vE_infty(A=AAr, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er_min=Er_min, Er_max=Er_max, vE=vE) for vE in vE_year_avg])
      rate_year_perp = np.mean([integrated_rate_perp_vE_infty(A=AAr, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er_min=Er_min, Er_max=Er_max, vE=vE) for vE in vE_year_avg])
    else:
      rate_jun_par = np.mean([integrated_rate_dcospsi_par_vE_infty(A=AAr, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er_min=Er_min, Er_max=Er_max, vE=vE) for vE in vE_jun_avg])
      rate_jun_perp = np.mean([integrated_rate_dcospsi_perp_vE_infty(A=AAr, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er_min=Er_min, Er_max=Er_max, vE=vE) for vE in vE_jun_avg])

      rate_dec_par = np.mean([integrated_rate_dcospsi_par_vE_infty(A=AAr, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er_min=Er_min, Er_max=Er_max, vE=vE) for vE in vE_dec_avg])
      rate_dec_perp = np.mean([integrated_rate_dcospsi_perp_vE_infty(A=AAr, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er_min=Er_min, Er_max=Er_max, vE=vE) for vE in vE_dec_avg])

      rate_year_par = np.mean([integrated_rate_dcospsi_par_vE_infty(A=AAr, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er_min=Er_min, Er_max=Er_max, vE=vE) for vE in vE_year_avg])
      rate_year_perp = np.mean([integrated_rate_dcospsi_perp_vE_infty(A=AAr, rhoD=rhoD, Md=Md, Mt=Mt, sigma0=sigma0, v0=v0, Er_min=Er_min, Er_max=Er_max, vE=vE) for vE in vE_year_avg])

    print(f'{Er_over_E0_r_min:.1f}-{Er_over_E0_r_max:.1f}, {rate_jun_par/Rzero:.3f}, {rate_jun_perp/Rzero:.3f}, {rate_dec_par/Rzero:.3f}, {rate_dec_perp/Rzero:.3f}, {rate_year_par/Rzero:.3f}, {rate_year_perp/Rzero:.3f}')


 #qrn = np.linspace(0, 10, 1000)
  qrn = qrn(A=AAr_nounits, a_n=a_n, b_n=b_n, Er=Er)
  q = momentum_transfer(Mt=Mt, Er=Er)
  thin_shell_approx = approx_form_factor_squared(qrn=qrn, alpha=1/3.)
  thin_shell = np.power(thin_shell_form_factor(qrn=qrn), 2)
  SD_Engel = approx_SD_form_factor_squared(qrn=qrn, A=AAr_nounits)
  plt.plot(qrn/hbar, thin_shell, label='thin shell (Eq. 4.2)')
  plt.plot(qrn/hbar, thin_shell_approx, label='Gaussian scatterer (Eq. 4.4 with $\\alpha=1/3$)')
  plt.plot(qrn/hbar, SD_Engel, label='SD approximation (Eq. 4.5)')
  plt.legend()
  plt.xlabel('$qr_n$')
  plt.ylabel('$F^2$')
 #plt.xlim(0, 10)
 #plt.ylim(1e-4, 1e0)
  plt.yscale('log')
  plt.title(f'Fig. 3 for A={AAr_nounits}')
  plt.show()

  solid_sphere = np.power(solid_sphere_form_factor(qrn=qrn), 2)
  solid_sphere_approx = approx_form_factor_squared(qrn=qrn, alpha=1/5.)
  helm = np.power(helm_form_factor(q=q, A=AAr_nounits, a=a, s=s), 2)
  helm_simpler = np.power(helm_form_factor_simpler(q=q, A=AAr_nounits, s=s, a_n=a_n, b_n=b_n), 2)
  plt.plot(qrn/hbar, solid_sphere, label='solid sphere (Eq. 4.3)')
  plt.plot(qrn/hbar, solid_sphere_approx, label='Eq. 4.4 with $\\alpha=1/5$')
  plt.plot(qrn/hbar, helm, label=f'Helm form factor (Eq. 4.7 and 4.11 with s={s/nu.fm:.2f} fm, a={a/nu.fm:.2f} fm)')
  plt.plot(qrn/hbar, helm_simpler, label=f'Helm form factor (Eq. 4.7 with s={s/nu.fm:.2f} fm, $r_n={a_n/nu.fm:.2f} fm * A^{{1/3}} + {b_n/nu.fm:.2f} fm$)')
  plt.legend()
  plt.xlabel('$qr_n$')
  plt.ylabel('$F^2$')
 #plt.xlim(0, 10)
 #plt.ylim(1e-4, 1e0)
  plt.yscale('log')
  plt.title(f'Fig. 4 for A={AAr_nounits}')
  plt.show()

  A = np.arange(0, 250)
  plt.plot(A, np.sqrt(r_n_squared(A=A, a=a, s=s)), label=f'$r_n$ from Eq. 4.11 (s={s/nu.fm:.2f} fm, a={a/nu.fm:.2f} fm)')
  plt.plot(A, r_n_approx(A=A, a_n=a_n, b_n=b_n), label=f'$r_n={a_n/nu.fm:.2f} A^{{1/3}}$ fm + {b_n/nu.fm:.2f} fm')
  plt.legend()
  plt.xlabel('A')
  plt.ylabel('$r_n$ [fm]')
  plt.title('Cross-check of $r_n$ approximation')
  plt.show()

  # r_rms_square_engler^2 = 0.93^2 * A^(2/3)
  #                       = 3/5 * r_n^2 + 3 * s^2
  # without solving, we instead use approximated values from the text
  s_engler = 1.0 * nu.fm
  a_n_engler = 1.0 * nu.fm # 0.89 fm
  b_n_engler = 0.0 * nu.fm # 0.30 fm
  helm_simpler_engler = np.power(helm_form_factor_simpler(q=q, A=AAr_nounits, s=s_engler, a_n=a_n_engler, b_n=b_n_engler), 2)
  q_in_inverse_fm = q / hbar * nu.fm # convert momentum from MeV to 1/fm
  plt.plot(q_in_inverse_fm, helm, label=f'Helm form factor (Eq. 4.7 and 4.11 with s={s/nu.fm:.2f} fm, a={a/nu.fm:.2f} fm)')
  plt.plot(q_in_inverse_fm, helm_simpler, label=f'Helm form factor (Eq. 4.7 with s={s/nu.fm:.2f} fm, $r_n={a_n/nu.fm:.2f} fm * A^{{1/3}} + {b_n/nu.fm:.2f} fm$)')
  plt.plot(q_in_inverse_fm, helm_simpler_engler, label=f'Helm form factor (Eq. 4.7 with s={s_engler/nu.fm:.2f} fm, $r_n={a_n_engler/nu.fm:.2f} fm * A^{{1/3}} + {b_n_engler/nu.fm:.2f} fm$)')
  plt.legend()
  plt.xlabel('$q$ [1/fm]')
  plt.ylabel('$F^2$')
 #plt.xlim(0, 10)
 #plt.ylim(1e-4, 1e0)
  plt.yscale('log')
  plt.title(f'Fig. 6 for A={AAr_nounits}')
  plt.show()

  plt.plot(Er/nu.keV, helm, label=f'Helm form factor (Eq. 4.7 and 4.11 with s={s/nu.fm:.2f} fm, a={a/nu.fm:.2f} fm)')
  plt.plot(Er/nu.keV, helm_simpler, label=f'Helm form factor (Eq. 4.7 with s={s/nu.fm:.2f} fm, $r_n={a_n/nu.fm:.2f} fm * A^{{1/3}} + {b_n/nu.fm:.2f} fm$)')
  plt.plot(Er/nu.keV, helm_simpler_engler, label=f'Helm form factor (Eq. 4.7 with s={s_engler/nu.fm:.2f} fm, $r_n={a_n_engler/nu.fm:.2f} fm * A^{{1/3}} + {b_n_engler/nu.fm:.2f} fm$)')
  plt.legend()
  plt.xlabel('$E_R$ [keV]')
  plt.ylabel('$F^2$')
 #plt.xlim(0, 10)
 #plt.ylim(1e-4, 1e0)
  plt.yscale('log')
  plt.title(f'Fig. 8 for A={AAr_nounits}')
  plt.show()

  plt.plot(Er/nu.keV, rate_year_avg/dru, label=f'R_0 S(E)')
  plt.plot(Er/nu.keV, rate_year_avg/dru * helm, label=f'R_0 S(E) $F^2$')
  plt.legend()
  plt.xlabel('$E_R$ [keV]')
  plt.ylabel('rate [events/(kg day keV)]')
 #plt.xlim(0, 10)
 #plt.ylim(1e-4, 1e0)
  plt.yscale('log')
  plt.title(f'Overall spectrum')
  plt.show()
