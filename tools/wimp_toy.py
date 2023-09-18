import numericalunits as nu
import wimp_rate
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.special import erf
import json

class NRAcceptance:
  ''' Provides acceptance functions as a function of energy and S1
      as estimated from different models:
       - full_acceptance_ene
       - full_acceptance_S1
  '''
  def __init__(self, toy, source, emin=0, emax=4000, estep=0.01):
    self._E = np.arange(emin, emax, step=estep)

    if source == 'DS50_highmass_empirical':
      # parameterised vs Enr from Fig. 10 from https://arxiv.org/pdf/1802.07198.pdf
      f90_acceptance_ene = lambda x: 0.49*erf((x-63)/7.7254)+0.49
      sel_acceptance_ene = lambda x: 0.37*erf((x-8)/1)+0.37

      # derive parameterisations vs S1
      _f90 = f90_acceptance_ene(self._E)
      _sel = sel_acceptance_ene(self._E)
      _S1  = toy.get_nr_visene(self._E) * toy.LY
      f90_acceptance_S1 = interpolate.interp1d(_S1, _f90)
      sel_acceptance_S1 = interpolate.interp1d(_S1, _sel)

      # overall acceptance
      self.full_acceptance_ene = lambda x: f90_acceptance_ene(x) * sel_acceptance_ene(x)
      self.full_acceptance_S1  = lambda x: f90_acceptance_S1(x)  * sel_acceptance_S1(x)
    elif source == 'DS50_highmass_interpolated':
      # use brute-force interpolation of Fig. 10 from https://arxiv.org/pdf/1802.07198.pdf
      _overall_nr_vs_ene_x, _overall_nr_vs_ene_y = np.loadtxt(f'{toy.datadir}/ds50_highmass_overall_nr_acceptance_vs_Enr.dat', unpack=True)
      _overall_nr_vs_S1_x, _overall_nr_vs_S1_y = np.loadtxt(f'{toy.datadir}/ds50_highmass_overall_nr_acceptance_vs_S1.dat', unpack=True)

      # overall acceptance
      self.full_acceptance_ene = interpolate.interp1d(_overall_nr_vs_ene_x, _overall_nr_vs_ene_y)
      self.full_acceptance_S1  = interpolate.interp1d(_overall_nr_vs_S1_x, _overall_nr_vs_S1_y)
    else:
      raise ValueError(source)

class WIMPToy:
  def __init__(self, emin, emax, estep, LY, acceptance_model, analysis_info, W=0.0195, nr_alpha=1, datadir='../data'):
    self.emin, self.emax, self.estep = emin, emax, estep
    self.LY = LY # light yield (ph/keV)
    self.W = W # effective work function (keV)
    self.nr_alpha = nr_alpha # excitation-to-ionisation ratio (1 for NR, 0.21 for ER)
    self.datadir = datadir
    with open(f'{self.datadir}/{analysis_info}') as infile:
        self.analysis_info = json.load(infile)

    self.wimprates = wimp_rate.WimpNR()

    # quenching factor
    _lx, _ly = np.loadtxt(f'{self.datadir}/leff.dat', unpack=True)
    self.leff = interpolate.interp1d(_lx,_ly)

    # recombination probability
    _rx, _ry = np.loadtxt(f'{self.datadir}/nr_reco.dat',  unpack=True)
    self.reco = interpolate.interp1d(_rx,_ry)

    # overall acceptance function vs Enr and S1 (f90 cut + other cuts)
    self.acceptance = NRAcceptance(self, acceptance_model, 0, 4000, 0.01)

  def get_nr_visene(self, ene):
    ''' Get NR visible energy
        https://arxiv.org/pdf/1707.05630.pdf
    '''
    qene   = ene*self.leff(ene) # apply quenching factor
    qene   = np.where(qene>self.W, qene, 0)
    visene = qene * (self.nr_alpha + self.reco(qene)) / (1 + self.nr_alpha)
    return visene

  def get_S1(self, visene):
    ''' Convert visible energy into number of photoelectrons, applying
        Poisson fluctuations
    '''
    npe    = self.LY*visene
    npe    = np.random.poisson(npe)
    return npe

  def get_events(self, N_events, Mw, xsec, exposure):
    ene = np.arange(self.emin, self.emax, self.estep)
    spectrum = self.wimprates.rate_wimp_nr(erec=ene, mw=Mw, sigma_nucleon=xsec)

    norm = np.sum(spectrum)

    E = np.random.choice(ene, N_events, p=spectrum/norm)
    weight = np.ones_like(E) * norm * exposure * self.estep / np.shape(E)[0]
    S1 = self.get_S1(self.get_nr_visene(E))

    acceptance = self.acceptance.full_acceptance_S1(S1) # calculated on each event based on its S1
    mask = np.random.binomial(1, acceptance).astype(bool) # events to keep

    return (E, S1, weight*acceptance)

if __name__ == '__main__':
  import sys
  sys.path.append('./tools')
  
  toy = WIMPToy(emin=0, emax=500, estep=0.01, LY=8, acceptance_model='DS50_highmass_interpolated', W=0.0195, nr_alpha=1, datadir='data')

  exposure = 16660/1000./365 # ton yr

  E, S1, weight = toy.get_events(N_events=100000, Mw=100, xsec=1.14e-44, exposure=exposure)

  fig, ax = plt.subplots(2, 1)
  ax = ax.ravel()
  h_S1 = ax[0].hist(S1, weights=weight, bins=100)
  ax[0].set_xlabel('S1')
  h_E = ax[1].hist(E, weights=weight, bins=100)
  ax[1].set_xlabel('E [keV]')
  plt.show()
  print(f'Total event yield for {exposure} t yr: {np.sum(h_S1[0])} to compare with {np.sum(weight)}')
