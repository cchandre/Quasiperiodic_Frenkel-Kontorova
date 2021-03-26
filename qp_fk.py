import numpy as xp
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from functools import partial
from scipy.io import savemat
import time
from datetime import date
import warnings
warnings.filterwarnings("ignore")

N = 2 ** 10
alpha = xp.array([1.246979603717467, 2.801937735804838])

N1 = 1024
N2 = 1024
eps1 = xp.linspace(0.007, 0.014, N1)
eps2 = xp.linspace(0.0025, 0.0038, N2)
potential = 'pot1'

threshold = 1e-7
TolMax = 1e5
TolMin = 1e-5

dim = len(alpha)
zero_ = dim * (0,)
ind_nu = dim * (xp.fft.fftfreq(N, d=1/N),)
nu = xp.meshgrid(*ind_nu, indexing='ij')
alpha_nu = 2.0 * xp.pi * xp.einsum('i,i...->...', alpha, nu)
exp_alpha_nu = xp.exp(1j * alpha_nu)
lk_alpha_nu = 2.0 * (xp.cos(alpha_nu) - 1.0)
ilk_alpha_nu = xp.divide(1.0, lk_alpha_nu, where=lk_alpha_nu!=0)
sml_div = exp_alpha_nu - 1.0
sml_div = xp.divide(1.0, sml_div, where=sml_div!=0)
ind_phi = dim * (2.0 * xp.pi * xp.arange(0, N) / N,)
phi = xp.meshgrid(*ind_phi, indexing='ij')
threshold *= N**dim

def dv(phi_, eps):
	if potential == 'pot1':
		return eps[0] * alpha[0] * xp.sin(phi_[0]) + eps[1] * alpha[1] * xp.sin(phi_[1])
	elif potential == 'pot2':
		return eps[0] * (alpha[0] + alpha[1]) * xp.sin(2.0 * phi_[0]+ 2.0 * phi_[1]) + eps[1] * (alpha[0] * xp.sin(phi_[0]) + alpha[1] * xp.sin(phi_[1]))

def refine_h(h, lam, eps):
	fft_h = xp.fft.fftn(h)
	fft_h[xp.abs(fft_h) <= threshold] = 0.0
	h_thresh = xp.fft.ifftn(fft_h)
	arg_v = phi + 2.0 * xp.pi * xp.tensordot(alpha, h_thresh, axes=0)
	l = 1.0 + xp.fft.ifftn(1j * alpha_nu * fft_h)
	epsilon = xp.fft.ifftn(lk_alpha_nu * fft_h) + lam + dv(arg_v, eps)
	fft_l_eps = xp.fft.fftn(l * epsilon)
	fft_l = xp.fft.fftn(l)
	delta = - fft_l_eps[zero_] / fft_l[zero_]
	w = xp.fft.ifftn((delta * fft_l + fft_l_eps) * sml_div)
	ll = l * xp.fft.ifftn(xp.fft.fftn(l) * exp_alpha_nu.conj())
	fft_wll = xp.fft.fftn(w / ll)
	fft_ill = xp.fft.fftn(1.0 / ll)
	w0 = - fft_wll[zero_] / fft_ill[zero_]
	del_l = xp.fft.ifftn((fft_wll + w0 * fft_ill) * sml_div.conj())
	h = xp.real(h_thresh + del_l * l)
	lam = xp.real(lam + delta)
	arg_v = phi + 2.0 * xp.pi * xp.tensordot(alpha, h, axes=0)
	err = xp.abs(xp.fft.ifftn(lk_alpha_nu * xp.fft.fftn(h)) + lam + dv(arg_v, eps)).max()
	return h, lam, err

def converge_point(eps1, eps2):
	h = - xp.fft.ifftn(xp.fft.fftn(dv(phi, xp.array([eps1, eps2]))) * ilk_alpha_nu)
	lam = 0.0
	err = 1.0
	it_count = 0
	while TolMax >= err >= TolMin:
		h, lam, err = refine_h(h, lam, xp.array([eps1, eps2]))
		it_count += 1
	if err <=TolMin:
		it_count = 0
	return [(err <= TolMin), it_count]

def save_data(name, data, timestr):
	mdic = dict({'alpha': alpha, 'N': N, 'threshold': threshold / N**dim, 'TolMin': TolMin, 'TolMax': TolMax, 'eps1': eps1, 'eps2': eps2, 'potential': potential})
	mdic.update({'data': data})
	today = date.today()
	date_today = today.strftime(" %B %d, %Y\n")
	email = 'cristel.chandre@univ-amu.fr'
	mdic.update({'date': date_today, 'author': email})
	savemat(name + '_' + timestr + '.mat', mdic)

def main():
	timestr = time.strftime("%Y%m%d_%H%M")
	num_cores = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(num_cores)
	data = []
	for epsilon2 in tqdm(eps2):
		converge_point_ = partial(converge_point, eps2=epsilon2)
		for result in pool.imap(converge_point_, iterable=eps1):
			data.append(result)
		save_data('qpFK_converge_region', data, timestr)
	data = xp.array(data).reshape((N1, N2, 2))
	save_data('qpFK_converge_region', data, timestr)
	plt.pcolor(data[:, :, 0])
	plt.show()

if __name__ == "__main__":
	main()
