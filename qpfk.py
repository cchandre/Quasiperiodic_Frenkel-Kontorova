import numpy as xp
from numpy import linalg as LA
from numpy.fft import fftn, ifftn, fftshift, ifftshift
from scipy.optimize import root
import matplotlib.pyplot as plt
from qpfk_modules import compute_line_norm, compute_region
from qpfk_dict import dict
import warnings
warnings.filterwarnings("ignore")

def main():
    case = qpFK(dict)
    eval(case.Method + '(case)')
    plt.show()

class qpFK:
    def __repr__(self):
        return '{self.__class__.__name__}({self.DictParams})'.format(self=self)

    def __str__(self):
        return 'Quasiperiodic Frenkel-Kontorova ({self.__class__.__name__}) model with alpha = {self.alpha} and omega = {self.omega}'.format(self=self)

    def __init__(self, dict):
        for key in dict:
            setattr(self, key, dict[key])
        self.DictParams = dict
        self.dim = len(self.alpha)
        self.zero_ = self.dim * (0,)
        self.id = xp.reshape(xp.identity(self.dim), 2 * (self.dim,) + self.dim * (1,))

    def set_var(self, L):
        ind_nu = self.dim * (xp.hstack((xp.arange(0, L // 2), xp.arange(- L // 2, 0))),)
        ind_phi = self.dim * (xp.linspace(0.0, 2.0 * xp.pi, L, endpoint=False, dtype=self.Precision),)
        nu = xp.meshgrid(*ind_nu, indexing='ij')
        self.phi = xp.meshgrid(*ind_phi, indexing='ij')
        self.Omega = 2.0 * xp.pi * self.alpha
        self.Omega_nu = xp.einsum('i,i...->...', self.Omega, nu)
        if hasattr(self, 'alpha_perp'):
            self.Omega_perp_nu = 2.0 * xp.pi * xp.einsum('i,i...->...', self.alpha_perp, nu)
        self.exp_Omega_nu = xp.exp(1j * self.omega * self.Omega_nu)
        self.sml_div = self.exp_Omega_nu - 1.0
        self.sml_div = xp.divide(1.0, self.sml_div, where=self.sml_div!=0)
        self.sml_div[self.zero_] = 0.0
        self.lk = 2.0 * (xp.cos(self.omega * self.Omega_nu) - 1.0)
        self.ilk = xp.divide(1.0, self.lk, where=self.lk!=0)
        self.ilk[self.zero_] = 0.0
        self.rescale_fft = self.Precision(L ** self.dim)
        self.tail_indx = self.dim * xp.index_exp[L//4:3*L//4+1]
        self.pad = self.dim * ((L//4, L//4),)

    def initial_h(self, eps, L, method='one_step'):
        self.set_var(L)
        if method == 'zero':
            return [xp.zeros_like(self.lk), 0.0]
        elif method == 'one_step':
            return [- ifftn(self.fft_h(self.Dv(self.phi, eps, self.Omega)) * self.ilk).real, 0.0]
        else:
            h = - ifftn(self.fft_h(self.Dv(self.phi, eps, self.Omega)) * self.ilk).real
            sol = root(self.conjug_eq, h.flatten(), args=(eps, L), method=method, options={'fatol': 1e-9})
            if sol.success:
                return [sol.x.reshape((L, L)), 0.0]
            else:
                return [h, 0.0]

    def conjug_eq(self, h, eps, L):
        arg_v = (self.phi + xp.tensordot(self.Omega, h.reshape(self.dim * (L,)), axes=0)) % (2.0 * xp.pi)
        return (ifftn(self.lk * self.fft_h(h.reshape(self.dim * (L,)))).real + self.Dv(arg_v, eps, self.Omega)).flatten()

    def refine_h(self, h, lam, eps):
        self.set_var(h.shape[0])
        fft_h = self.fft_h(h)
        arg_v = (self.phi + xp.tensordot(self.Omega, h, axes=0)) % (2.0 * xp.pi)
        fft_l = 1j * self.Omega_nu * fft_h
        fft_l[self.zero_] = self.rescale_fft
        l = ifftn(fft_l).real
        epsilon = ifftn(self.lk * fft_h).real + self.Dv(arg_v, eps, self.Omega) + lam
        fft_leps = fftn(l * epsilon)
        delta = - fft_leps[self.zero_].real / self.rescale_fft
        w = ifftn((delta * fft_l + fft_leps) * self.sml_div).real
        ll = l * ifftn(fft_l * self.exp_Omega_nu.conj()).real
        fft_wll = fftn(w / ll)
        fft_ill = fftn(1.0 / ll)
        w0 = - fft_wll[self.zero_].real / fft_ill[self.zero_].real
        beta = ifftn((fft_wll + w0 * fft_ill) * self.sml_div.conj()).real
        fft_h = self.fft_h(h + beta * l - xp.mean(beta * l) * l)
        lam_ = lam + delta
        tail_norm = xp.abs(fft_h[self.tail_indx]).max() / self.rescale_fft
        if self.AdaptSize and (tail_norm >= self.Threshold) and (h.shape[0] < self.Lmax):
            self.set_var(2 * h.shape[0])
			fft_h = ifftshift(xp.pad(fftshift(self.fft_h(h)), self.pad)) * (2 ** self.dim)
			lam_ = lam
        h_ = ifftn(fft_h).real
        arg_v = (self.phi + xp.tensordot(self.Omega, h_, axes=0)) % (2.0 * xp.pi)
        err = xp.abs(ifftn(self.lk * fft_h).real + self.Dv(arg_v, eps, self.Omega) + lam_).max()
        if self.AdaptSize and (tail_norm >= self.Threshold) and (h.shape[0] >= self.Lmax):
			err = self.TolMax ** 2
        if self.MonitorGrad:
            dh_ = self.id + xp.tensordot(self.Omega, xp.gradient(h_, 2.0 * xp.pi / self.Precision(h_.shape[0])), axes=0)
            det_h_ = xp.abs(LA.det(xp.moveaxis(dh_, [0, 1], [-2, -1]))).min()
            if det_h_ <= self.TolMin:
                print('\033[31m        warning: non-invertibility...\033[00m')
        return h_, lam_, err

    def fft_h(self, h):
        fft_h = fftn(h)
        fft_h[self.zero_] = 0.0
        fft_h[xp.abs(fft_h) <= self.Threshold * xp.abs(fft_h).max()] = 0.0
        return fft_h

    def pad_h(self, h):
        return ifftn(ifftshift(xp.pad(fftshift(self.fft_h(h)), self.pad))).real * (2 ** self.dim)

    def norms(self, h, r=0):
        fft_h = self.fft_h(h)
        if hasattr(self, 'alpha_perp'):
            return [xp.sqrt((xp.abs(ifftn(self.Omega_nu ** r * fft_h)) ** 2).sum()), xp.sqrt((xp.abs(ifftn(self.Omega_perp_nu ** r * fft_h)) ** 2).sum())]
        else:
            return xp.sqrt((xp.abs(ifftn(self.Omega_nu ** r * fft_h)) ** 2).sum())

if __name__ == "__main__":
    main()
