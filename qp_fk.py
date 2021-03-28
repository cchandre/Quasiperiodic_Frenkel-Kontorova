import numpy as xp
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
		'eps_lim': [[0.007, 0.0025], [0.014,  0.0038]],
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
	timestr = time.strftime("%Y%m%d_%H%M")
	num_cores = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(num_cores)
	data = []
	eps_vecs = xp.linspace(eps_lim, n_eps)
	for eps2 in tqdm(eps_vecs[1]):
		converge_point_ = lambda eps1: converge_point(eps1, eps2=eps2, case=case)
		for result in pool.imap(converge_point_, iterable=eps_vecs[0]):
			data.append(result)
		save_data('qpFK_converge_region', data, timestr, case)
	data = xp.array(data).reshape((case.n_eps, case.n_eps, 2))
	self.save_data('qpFK_converge_region', data, timestr, case)
	plt.pcolor(data[:, :, 0])
	plt.show()

class qpFK:
	def __init__(self, dv, dict_params):
		for key in dict_param:
            setattr(self, key, dict_params[key])
		self.DictParams = dict_params
		self.Precision = {32: xp.float32, 64: xp.float64, 128: xp.float128}.get(self.Precision, xp.float64)()
		self.dv = dv
		dim = len(self.alpha)
		self.alpha = xp.asarray(self.alpha, dtype=self.Precision)
		self.zero_ = dim * (0,)
		ind_nu = dim * (xp.fft.fftfreq(self.N, d=1/self.N),)
		nu = xp.meshgrid(*ind_nu, indexing='ij')
		self.alpha_nu = 2.0 * xp.pi * xp.einsum('i,i...->...', self.alpha, nu)
		self.exp_alpha_nu = xp.exp(1j * self.alpha_nu)
		self.lk_alpha_nu = 2.0 * (xp.cos(self.alpha_nu) - 1.0)
		self.sml_div = self.exp_alpha_nu - 1.0
		self.sml_div = xp.divide(1.0, self.sml_div, where=self.sml_div!=0)
		ind_phi = dim * (2.0 * xp.pi * xp.linspace(0.0, 1.0, self.N, endpoint=False),)
		self.phi = xp.meshgrid(*ind_phi, indexing='ij')
		self.threshold *= self.N**dim
		ilk_alpha_nu = xp.divide(1.0, self.lk_alpha_nu, where=self.lk_alpha_nu!=0)
		self.initial_h = lambda eps: - xp.fft.ifftn(xp.fft.fftn(self.dv(self.phi, eps)) * ilk_alpha_nu)

	def __repr__(self):
        return '{self.__class__.name__}({self.dv, self.DictParams})'.format(self=self)

    def __str__(self):
        return 'Quasiperiodic Frenkel-Kontorova ({self.__class__.name__}) model with alpha = {self.alpha}'.format(self=self)

	def refine_h(self, h, lam, eps):
		fft_h = xp.fft.fftn(h)
		fft_h[xp.abs(fft_h) <= self.threshold] = 0.0
		h_thresh = xp.fft.ifftn(fft_h)
		arg_v = self.phi + 2.0 * xp.pi * xp.tensordot(xp.asarray(self.alpha), h_thresh, axes=0)
		l = 1.0 + xp.fft.ifftn(1j * self.alpha_nu * fft_h)
		epsilon = xp.fft.ifftn(self.lk_alpha_nu * fft_h) + lam + self.dv(arg_v, eps)
		fft_l_eps = xp.fft.fftn(l * epsilon)
		fft_l = xp.fft.fftn(l)
		delta = - fft_l_eps[self.zero_] / fft_l[self.zero_]
		w = xp.fft.ifftn((delta * fft_l + fft_l_eps) * self.sml_div)
		ll = l * xp.fft.ifftn(xp.fft.fftn(l) * self.exp_alpha_nu.conj())
		fft_wll = xp.fft.fftn(w / ll)
		fft_ill = xp.fft.fftn(1.0 / ll)
		w0 = - fft_wll[self.zero_] / fft_ill[self.zero_]
		del_l = xp.fft.ifftn((fft_wll + w0 * fft_ill) * self.sml_div.conj())
		h = xp.real(h_thresh + del_l * l)
		lam = xp.real(lam + delta)
		arg_v = self.phi + 2.0 * xp.pi * xp.tensordot(xp.asarray(self.alpha), h, axes=0)
		err = xp.abs(xp.fft.ifftn(self.lk_alpha_nu * xp.fft.fftn(h)) + lam + self.dv(arg_v, eps)).max()
		return h, lam, err

def save_data(name, data, timestr, case):
	mdic = case.DictParams
	mdic.update({'data': data})
	today = date.today()
	date_today = today.strftime(" %B %d, %Y\n")
	email = 'cristel.chandre@univ-amu.fr'
	mdic.update({'date': date_today, 'author': email})
	savemat(name + '_' + timestr + '.mat', mdic)

def converge_point(eps1, eps2, case):
	h = case.initial_h([eps1, eps2])
	lam = 0.0
	err = 1.0
	it_count = 0
	while case.TolMax >= err >= case.TolMin:
		h, lam, err = case.refine_h(h, lam, [eps1, eps2])
		it_count += 1
	if err <= case.TolMin:
		it_count = 0
	return [(err <= case.TolMin), it_count]

if __name__ == "__main__":
	main()
