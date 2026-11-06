import logging
import numpy as np
import pymc as pm
import ray

class TGrid:
  def __init__(self, N=512, T_scale=1):
    self.N=N
    self.T_scale = T_scale

  def get_gridpt(T1, T2):
    return (
      round((T1 / self.T_scale) * self.N),
      round((T2 / self.T_scale) * self.N),
    )

  def symmetrize_gridpt(i,j):
    # use symmetry transformations to map all equal-physics gridpoints 
    # to the same single gridpoint.
    ...

  def get_symm_gridpt(T1,T2) -> list(int): # returns [i,j] a point in the T-grid
    return (
      self.symmetrize_gridpt(
        self.get_gridpt(T1, T2)
      )
    )

@ray.remote # paralellize
def run_mcmc(i: int, j: int) -> str:
  ... # run, save, return save path

class mcmc_manager:
  def __init__(self, n_batch, log=False):
    self.n_batch = n_batch
    self.log = log
    self.logger = logging.getLogger('mcmc_manager')
    self.logger.setLevel(logging.INFO)
    #
    # init:
    self.n_curr = 0
    self.chains_runned = {} # {(i,j): chain output loc (str)}
    self.chains_to_run = [] # [i,j]

  def add_event(self, i, j):
    if (i,j) in self.chains_runned.keys():
      return
    else:
      self.chains_to_run.append([i,j])
      self.n_curr += 1

  def dispatch(self):
    # start running all mcmc chains collected in parallel
    # once n_curr == n_batch
    chain_locs = ray.get([run_mcmc.remote(i,j) for [i,j] in self.chains_to_run])
    for pair, idx in enumerate(self.chains_to_run):
      i,j = pair
      self.chains_runned[(i,j)] = chain_locs[idx]
    #
    if self.log:
      self.logger.info(f"dispatch: Finished running {self.n_curr} MCMC events.")
    self.n_curr = 0
    self.chains_to_run = []
  
  def collect_batch(
    tvals: np.ndarray, # space-flattened array of (T1, T2) pairs for each x-y point
    t_grid: TGrid,
    start_idx: int = 0,
  ):
    idx = start_idx
    while (self.n_curr < self.n_batch) and idx < len(tvals):
      t1, t2 = tvals[idx]
      i,j = t_grid.get_symm_gridpt(t1, t2)
      self.add_event(i,j)
      #
      idx+=1
    if self.log:
      self.logger.info(
        f"collect_batch: added {self.n_curr} T-grid points to evaluate:\n"
        f"{self.chains_to_run}"
      )
    return idx

























