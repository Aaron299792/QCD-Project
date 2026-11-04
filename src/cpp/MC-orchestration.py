import numpy as np

class TGrid:
	def __init__(self, N=512, T_scale=1):
		self.N=N
		self.T_scale = T_scale
		self.mcmc_initialized = np.zeros(shape=(self.N,self.N), dtype=np.uint8)
		...

	def get_gridpt(T1, T2):
		return (
			round((T1 / self.T_scale) * self.N),
			round((T2 / self.T_scale) * self.N),
		)

	def symmetrize_gridpt(i,j):
		# use symmetry transformations to map all equal-physics gridpoints 
		# to the same single gridpoint.
		...

	def get_symm_gridpt(T1,T2):
		return (
			self.symmetrize_gridpt(
				self.get_gridpt(T1, T2)
			)
		)

	def init_mcmc(T1, T2):
		i, j = get_symm_gridpt(T1,T2)
		self.mcmc_initialized[i,j] = np.uint8(1)
