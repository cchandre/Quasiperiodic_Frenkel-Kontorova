import numpy as xp
from numpy.fft import fftn, ifftn, fftfreq
import matplotlib.pyplot as plt
import warnings
import convergence as cv
warnings.filterwarnings("ignore")

def main():
	# dict_params = {
	# 	'n': 2 ** 9,
	# 	'omega': 1.0,
	# 	'alpha': [1.246979603717467, 2.801937735804838],
	# 	'alpha_perp': [2.801937735804838, -1.246979603717467],
	# 	'potential': 'pot1_2d'}
	# dict_params.update({
	# 	'eps_n': 256,
	# 	'eps_region': [[0.007, 0.0025], [0.014,  0.0038]],
	# 	'eps_point': [0.009, 0.0030],
	# 	'eps_dir': [0.001, 0.01, xp.pi/5]})
	dict_params = {
		'n': 2 ** 10,
		'omega': 0.618033988749895,
		'alpha': [1.0],
		'potential': 'pot1_1d'}
	dict_params.update({
		'eps_n': 128,
		'eps_region': [[0.0, 2.0], [-0.8,  0.8]],
		'eps_type': 'cartesian'})
	dict_params.update({
		'tolmax': 1e2,
		'tolmin': 1e-10,
		'tolinit': 1e-1,
		'threshold': 1e-12,
		'precision': 64,
		'save_results': True})
	alpha = dict_params['alpha']
	dv = {
		'pot1_1d': lambda phi, eps: - alpha[0] / (2.0 * xp.pi) * (eps[0] * xp.sin(phi[0]) + eps[1] / 2.0 * xp.sin(2.0 * phi[0])),
		'pot1_2d': lambda phi, eps: alpha[0] * eps[0] * xp.sin(phi[0]) + alpha[1] * eps[1] * xp.sin(phi[1]),
		'pot2_2d': lambda phi, eps: alpha[0] * (eps[0] * xp.sin(2.0 * phi[0]+ 2.0 * phi[1]) + eps[1] * xp.sin(phi[0]))\
		 	+ alpha[1] * (eps[0] * xp.sin(2.0 * phi[0]+ 2.0 * phi[1]) + eps[1] * xp.sin(phi[1]))
		}.get(dict_params['potential'], 'pot1_2d')
	case = qpFK(dv, dict_params)
	data = cv.region(xp.linspace(case.eps_region[0], case.eps_region[1], case.eps_n).transpose(), case)
	plt.pcolor(data[:, :, 0])
	plt.show()
	#cv.point(case.eps_point[0], case.eps_point[1], case, gethull=True)

class qpFK:
	def __repr__(self):
		return '{self.__class__.name__}({self.dv, self.DictParams})'.format(self=self)

	def __str__(self):
		return 'Quasiperiodic Frenkel-Kontorova ({self.__class__.name__}) model with alpha = {self.alpha} and omega = {self.omega}'.format(self=self)

	def __init__(self, dv, dict_params):
		for key in dict_params:
			setattr(self, key, dict_params[key])
		self.DictParams = dict_params
		self.precision = {32: xp.float32, 64: xp.float64, 128: xp.float128}.get(self.precision, xp.float64)
		self.dv = dv
		self.dim = len(self.alpha)
		self.alpha = xp.array(self.alpha, self.precision)
		self.zero_ = self.dim * (0,)
		ind_nu = self.dim * (fftfreq(self.n, d=1.0/self.precision(self.n)),)
		nu = xp.asarray(xp.meshgrid(*ind_nu, indexing='ij'), dtype=self.precision)
		self.alpha_nu = xp.einsum('i,i...->...', self.alpha, nu)
		if hasattr(self, 'alpha_perp'):
			self.alpha_perp = xp.array(self.alpha_perp, self.precision)
			self.alpha_perp_nu = xp.einsum('i,i...->...', self.alpha_perp, nu)
		self.exp_alpha_nu = xp.exp(1j * 2.0 * xp.pi * self.omega * self.alpha_nu)
		self.lk_alpha_nu = 2.0 * (xp.cos(2.0 * xp.pi * self.omega * self.alpha_nu) - 1.0)
		self.sml_div = self.exp_alpha_nu - 1.0
		self.sml_div = xp.divide(1.0, self.sml_div, where=self.sml_div!=0)
		ind_phi = self.dim * (xp.linspace(0.0, 2.0 * xp.pi, self.n, endpoint=False),)
		self.phi = xp.asarray(xp.meshgrid(*ind_phi, indexing='ij'), dtype=self.precision)
		self.threshold *= self.n**self.dim
		self.ilk_alpha_nu = xp.divide(1.0, self.lk_alpha_nu, where=self.lk_alpha_nu!=0)
		self.initial_h = lambda eps: [- ifftn(fftn(self.dv(self.phi, eps)) * self.ilk_alpha_nu), 0.0]

	def refine_h(self, h, lam, eps):
		fft_h = fftn(h)
		fft_h[xp.abs(fft_h) <= self.threshold] = 0.0
		h_thresh = ifftn(fft_h)
		arg_v = self.phi + 2.0 * xp.pi * xp.tensordot(self.alpha, h_thresh, axes=0)
		l = 1.0 + 2.0 * xp.pi * ifftn(1j * self.alpha_nu * fft_h)
		epsilon = ifftn(self.lk_alpha_nu * fft_h) + lam + self.dv(arg_v, eps)
		fft_leps = fftn(l * epsilon)
		fft_l = fftn(l)
		delta = - fft_leps[self.zero_] / fft_l[self.zero_]
		w = ifftn((delta * fft_l + fft_leps) * self.sml_div)
		ll = l * ifftn(xp.fft.fftn(l) * self.exp_alpha_nu.conj())
		fft_wll = fftn(w / ll)
		fft_ill = fftn(1.0 / ll)
		w0 = - fft_wll[self.zero_] / fft_ill[self.zero_]
		dell = ifftn((fft_wll + w0 * fft_ill) * self.sml_div.conj())
		h = xp.real(h_thresh + dell * l)
		lam = xp.real(lam + delta)
		arg_v = self.phi + 2.0 * xp.pi * xp.tensordot(self.alpha, h, axes=0)
		err = xp.abs(ifftn(self.lk_alpha_nu * fftn(h)) + lam + self.dv(arg_v, eps)).max()
		return h, lam, err

	def norms(self, h, r=0):
		if hasattr(self, 'alpha_perp'):
			return [xp.sqrt(xp.abs(ifftn(self.alpha_nu ** r * fftn(h)) ** 2).sum()),\
				xp.sqrt(xp.abs(ifftn(self.alpha_perp_nu ** r * fftn(h)) ** 2).sum())]
		else:
			return xp.sqrt(xp.abs(ifftn(self.alpha_nu ** r * fftn(h)) ** 2).sum())

if __name__ == "__main__":
	main()
