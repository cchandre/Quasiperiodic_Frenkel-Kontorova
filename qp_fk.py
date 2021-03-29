import numpy as xp
from numpy.fft import fftn, ifftn, fftfreq
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from scipy.io import savemat
import time
from datetime import date
import warnings
warnings.filterwarnings("ignore")

def main():
	dict_params = {
		'N': 2 ** 10,
		'alpha': [1.246979603717467, 2.801937735804838],
		'potential': 'pot1',
		'n_eps': 1024,
		'eps_region': [[0.007, 0.0025], [0.014,  0.0038]],
		'eps_point': [0.009, 0.0030],
		'eps_dir': [0.001, 0.01, xp.pi/5],
		'TolMax': 1e5,
		'TolMin': 1e-5,
		'threshold': 1e-7,
		'Precision': 64
		}
	alpha = dict_params['alpha']
	dv = {
		'pot1': lambda phi, eps: eps[0] * alpha[0] * xp.sin(phi[0]) + eps[1] * alpha[1] * xp.sin(phi[1]),
		'pot2': lambda phi, eps: eps[0] * (alpha[0] + alpha[1]) * xp.sin(2.0 * phi[0]+ 2.0 * phi[1])\
		 + eps[1] * (alpha[0] * xp.sin(phi[0]) + alpha[1] * xp.sin(phi[1]))
		}.get(dict_params['potential'], 'pot1')
	case = qpFK(dv, dict_params)
	converge_region(xp.linspace(self.eps_region, case.n_eps), case)
	converge_point(case.eps_point[0], case.eps_point[1], case, gethull=True)
	data = []
	for i, eps1 in enumerate(xp.linspace(eps_dir[0], eps_dir[1], n_eps)):
		data.appeconverge_point(eps1, eps1 * xp.tan(eps_dir[2]), case, getnorm=[True, 4])


class qpFK:
	def __init__(self, dv, dict_params):
		for key in dict_param:
            setattr(self, key, dict_params[key])
		self.DictParams = dict_params
		self.Precision = {32: xp.float32, 64: xp.float64, 128: xp.float128}.get(self.Precision, xp.float64)()
		self.dv = dv
		dim = len(self.alpha)
		self.alpha = xp.asarray(self.alpha, dtype=self.Precision)
		alpha_perp = xp.asarray([self.alpha[1], -self.alpha[0]], dtype=self.Precision)
		self.zero_ = dim * (0,)
		ind_nu = dim * (fftfreq(self.N, d=1/self.N),)
		nu = xp.meshgrid(*ind_nu, indexing='ij')
		self.alpha_nu = 2.0 * xp.pi * xp.einsum('i,i...->...', self.alpha, nu)
		self.alpha_perp_nu = 2.0 * xp.pi * xp.einsum('i,i...->...', alpha_perp, nu)
		self.exp_alpha_nu = xp.exp(1j * self.alpha_nu)
		self.lk_alpha_nu = 2.0 * (xp.cos(self.alpha_nu) - 1.0)
		self.sml_div = self.exp_alpha_nu - 1.0
		self.sml_div = xp.divide(1.0, self.sml_div, where=self.sml_div!=0)
		ind_phi = dim * (2.0 * xp.pi * xp.linspace(0.0, 1.0, self.N, endpoint=False),)
		self.phi = xp.meshgrid(*ind_phi, indexing='ij')
		self.threshold *= self.N**dim
		ilk_alpha_nu = xp.divide(1.0, self.lk_alpha_nu, where=self.lk_alpha_nu!=0)
		self.initial_h = lambda eps: - ifftn(fftn(self.dv(self.phi, eps)) * ilk_alpha_nu)

	def __repr__(self):
        return '{self.__class__.name__}({self.dv, self.DictParams})'.format(self=self)

    def __str__(self):
        return 'Quasiperiodic Frenkel-Kontorova ({self.__class__.name__}) model with alpha = {self.alpha}'.format(self=self)

	def refine_h(self, h, lam, eps):
		fft_h = fftn(h)
		fft_h[xp.abs(fft_h) <= self.threshold] = 0.0
		h_thresh = ifftn(fft_h)
		arg_v = self.phi + 2.0 * xp.pi * xp.tensordot(xp.asarray(self.alpha), h_thresh, axes=0)
		l = 1.0 + ifftn(1j * self.alpha_nu * fft_h)
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
		arg_v = self.phi + 2.0 * xp.pi * xp.tensordot(xp.asarray(self.alpha), h, axes=0)
		err = xp.abs(ifftn(self.lk_alpha_nu * fftn(h)) + lam + self.dv(arg_v, eps)).max()
		return h, lam, err

	def norms(self, h, r):
		return xp.sqrt(xp.abs(ifftn(self.alpha_nu ** r * fftn(h)) ** 2).sum()),\
			xp.sqrt(xp.abs(ifftn(self.alpha_perp_nu ** r * fftn(h)) ** 2).sum())


def save_data(name, data, timestr, case):
	case.DictParams.update({'data': data})
	date_today = date.today().strftime(" %B %d, %Y\n")
	case.DictParams.update({'date': date_today, 'author': 'cristel.chandre@univ-amu.fr'})
	savemat(name + '_' + timestr + '.mat', case.DictParams)

def converge_point(eps1, eps2, case, gethull=False, getnorm=[False, 0]):
	h = case.initial_h([eps1, eps2])
	lam = 0.0
	err = 1.0
	it_count = 0
	while case.TolMax >= err >= case.TolMin:
		h, lam, err = case.refine_h(h, lam, [eps1, eps2])
		it_count += 1
	if gethull:
		timestr = time.strftime("%Y%m%d_%H%M")
		save_data('qpFK_hull', h, timestr, case)
		return (err <= case.TolMin)
	if getnorm[0]:
		return (err <= case.TolMin), self.norm(h, getnorm[1])
	if err <= case.TolMin:
		it_count = 0
	return [(err <= case.TolMin), it_count]

def converge_region(eps_region, case):
	timestr = time.strftime("%Y%m%d_%H%M")
	num_cores = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(num_cores)
	data = []
	for eps2 in tqdm(eps_region[1]):
		converge_point_ = lambda eps1: converge_point(eps1, eps2=eps2, case=case)
		for result in pool.imap(converge_point_, iterable=eps_region[0]):
			data.append(result)
		save_data('qpFK_converge_region', data, timestr, case)
	data = xp.array(data).reshape((case.n_eps, case.n_eps, 2))
	save_data('qpFK_converge_region', data, timestr, case)
	plt.pcolor(data[:, :, 0])
	plt.show()

if __name__ == "__main__":
	main()
