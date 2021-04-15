import numpy as xp
from numpy.fft import fftn, ifftn, fftfreq
from numpy import linalg as LA
import convergence_reformat as cv
import copy
import warnings
warnings.filterwarnings("ignore")

def main():
	dict_params = {
		'n': 2 ** 6,
		'omega0': [0.618033988749895, -1.0],
		'N': [[1, 1], [1, 0]],
		'Omega': [1.0, 0.0],
		'potential': 'pot1_2d'}
	dict_params.update({
		'eps_n': 64,
		'eps_region': [[0.0, 0.05], [xp.pi/4,  xp.pi/4]],
		'eps_indx': [0, 1],
		'eps_type': 'polar'})
	# dict_params = {
	# 	'n': 2 ** 8,
	# 	'omega0': [1.324717957244746, 1.754877666246693, 1.0],
	# 	'Omega': [1.0, 1.0, -1.0],
	# 	'potential': 'pot1_3d'}
	# dict_params.update({
	# 	'eps_n': 256,
	# 	'eps_region': [[0.0, 0.15], [0.0,  xp.pi/2.0], [0.1, 0.1]],
	# 	'eps_indx': [0, 1],
	# 	'eps_type': 'polar'})
	dict_params.update({
		'renormalization': True,
		'tolmax': 1e5,
		'tolmin': 1e-7,
		'dist_surf': 1e-5,
		'maxiter': 500,
		'threshold': 1e-8,
		'precision': 64,
		'save_results': False,
		'plot_results': True})
	Omega = dict_params['Omega']
	dv = {
		'pot1_2d': lambda phi, eps: Omega[0] * eps[0] * xp.sin(phi[0]) + eps[1] * (Omega[0] + Omega[1]) * xp.sin(phi[0] + phi[1]),
		'pot1_3d': lambda phi, eps: - Omega[0] * eps[0] * xp.sin(phi[0]) - Omega[1] * eps[1] * xp.sin(phi[1]) - Omega[2] * eps[2] * xp.sin(phi[2])
		}.get(dict_params['potential'], 'pot1_2d')
	case = ConfKAM(dv, dict_params)
	data = cv.line(case.eps_region, case, method='critical', display=True)
	# data = cv.region(case)


class ConfKAM:
	def __repr__(self):
		return '{self.__class__.name__}({self.dv, self.DictParams})'.format(self=self)

	def __str__(self):
		return 'KAM in configuration space ({self.__class__.name__}) with omega0 = {self.omega0} and Omega = {self.Omega}'.format(self=self)

	def __init__(self, dv, dict_params):
		for key in dict_params:
			setattr(self, key, dict_params[key])
		self.DictParams = dict_params
		self.precision = {64: xp.float64, 128: xp.float128}.get(self.precision, xp.float64)
		self.dv = dv
		dim = len(self.omega0)
		self.N = xp.asarray(self.N, dtype=int)
		self.invN = LA.inv(self.N).transpose()
		eigenval, w_eig = LA.eig(self.N.transpose())
		self.Eigenvalue = xp.real(eigenval[xp.abs(eigenval) < 1])
		self.omega0 = xp.array(self.omega0, dtype=self.precision)
		self.zero_ = dim * (0,)
		ind_nu = dim * (fftfreq(self.n, d=1.0/self.precision(self.n)),)
		self.nu = xp.meshgrid(*ind_nu, indexing='ij')
		N_nu = xp.einsum('ij,j...->i...', self.N, self.nu)
		mask = xp.prod(abs(N_nu) <= self.n/2, axis=0, dtype=bool)
		self.nu_mask = []
		self.N_nu_mask = []
		for it in range(dim):
			self.nu_mask += (self.nu[it][mask].astype(int),)
			self.N_nu_mask += (N_nu[it][mask].astype(int),)
		self.omega0_nu = xp.einsum('i,i...->...', self.omega0, self.nu)
		self.lk = - self.omega0_nu ** 2
		self.sml_div = 1j * self.omega0_nu
		self.sml_div = xp.divide(1.0, self.sml_div, where=self.sml_div!=0)
		ind_phi = dim * (xp.linspace(0.0, 2.0 * xp.pi, self.n, endpoint=False, dtype=self.precision),)
		self.phi = xp.meshgrid(*ind_phi, indexing='ij')
		self.rescale_fft = self.precision(self.n ** dim)
		self.threshold *= self.rescale_fft
		self.ilk = xp.divide(1.0, self.lk, where=self.lk!=0)

	def initial_h(self, eps):
		h = - ifftn(fftn(self.dv(self.phi, eps)) * self.ilk)
		Omega = xp.array(self.Omega, dtype=self.precision)
		hull = Hull(self, h=h, lam=0.0, Omega=Omega, dv=self.dv)
		hull.Omega_nu = xp.einsum('i,i...->...', hull.Omega, self.nu)
		return hull

	def renorm_h(self, hull):
		hull_ = copy.deepcopy(hull)
		omega_ = (self.N.transpose()).dot(hull.Omega)
		n_omega_ = xp.sqrt((omega_ ** 2).sum())
		hull_.Omega = omega_ / n_omega_
		hull_.Omega_nu = xp.einsum('i,i...->...', hull_.Omega, self.nu)
		hull_.lam = hull.lam * n_omega_ / self.Eigenvalue ** 2
		hull_.dv = lambda phi, eps: self.dv(xp.einsum('ij,j...->i...', self.invN, phi), eps) * n_omega_ / self.Eigenvalue ** 2
		fft_h = fftn(hull.h)
		fft_h_ = xp.zeros_like(fft_h)
		fft_h_[self.nu_mask] = fft_h[self.N_nu_mask]
		hull_.h = xp.real(ifftn(fft_h_)) * n_omega_
		return hull_

	def refine_h(self, hull, eps):
		fft_h = fftn(hull.h)
		fft_h[xp.abs(fft_h) <= self.threshold] = 0.0
		h_thresh = ifftn(fft_h)
		arg_v = self.phi + xp.tensordot(hull.Omega, h_thresh, axes=0)
		fft_l = 1j * hull.Omega_nu * fft_h
		fft_l[self.zero_] = self.rescale_fft
		lfunc = ifftn(fft_l)
		epsilon = ifftn(self.lk * fft_h) + hull.lam + hull.dv(arg_v, eps)
		fft_leps = fftn(lfunc * epsilon)
		delta = - fft_leps[self.zero_] / fft_l[self.zero_]
		w = ifftn((delta * fft_l + fft_leps) * self.sml_div)
		fft_wll = fftn(w / lfunc ** 2)
		fft_ill = fftn(1.0 / lfunc ** 2)
		w0 = - fft_wll[self.zero_] / fft_ill[self.zero_]
		beta = ifftn((fft_wll + w0 * fft_ill) * self.sml_div.conj()) * lfunc
		hull_ = copy.deepcopy(hull)
		hull_.h = xp.real(h_thresh + beta - xp.mean(beta) * lfunc / xp.mean(lfunc))
		hull_.lam = xp.real(hull.lam + delta)
		arg_v = self.phi + xp.tensordot(hull.Omega, hull_.h, axes=0)
		err = xp.abs(ifftn(self.lk * fftn(hull_.h)) + hull_.lam + hull_.dv(arg_v, eps)).max()
		return hull_, err

	def image_h(self, hull, eps):
		if self.renormalization:
			hull_ = self.renorm_h(hull)
		else:
			hull_ = copy.deepcopy(hull)
		return self.refine_h(hull_, eps)

	def norms(self, hull, r=0):
		return [xp.sqrt(xp.abs(ifftn(self.omega0_nu ** r * fftn(hull.h)) ** 2).sum()), xp.sqrt(xp.abs(ifftn(hull.Omega_nu ** r * fftn(hull.h)) ** 2).sum())]


class Hull:
	def __init__(self, case, h=[], lam=[], Omega=[], dv=[]):
		self.h = h
		self.lam = lam
		self.Omega = Omega
		self.Omega_nu = []
		self.dv = dv


if __name__ == "__main__":
	main()
