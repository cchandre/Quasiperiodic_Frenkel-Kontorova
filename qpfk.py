import numpy as xp
from numpy import linalg as LA
from numpy.fft import fftn, ifftn, fftfreq, fftshift, ifftshift
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
        ind_nu = self.dim * (fftfreq(L, d=1.0/self.Precision(L)),)
        ind_phi = self.dim * (xp.linspace(0.0, 2.0 * xp.pi, L, endpoint=False, dtype=self.Precision),)
        nu = xp.meshgrid(*ind_nu, indexing='ij')
        self.phi = xp.meshgrid(*ind_phi, indexing='ij')
        self.alpha_nu = 2.0 * xp.pi * xp.einsum('i,i...->...', self.alpha, nu)
        if hasattr(self, 'alpha_perp'):
            self.alpha_perp_nu = xp.einsum('i,i...->...', self.alpha_perp, nu)
        self.exp_alpha_nu = xp.exp(1j * self.omega * self.alpha_nu)
        self.sml_div = self.exp_alpha_nu - 1.0
        self.sml_div = xp.divide(1.0, self.sml_div, where=self.sml_div!=0)
        self.sml_div[self.zero_] = 0.0
        self.lk = 2.0 * (xp.cos(self.omega * self.alpha_nu) - 1.0)
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
            return [- ifftn(fftn(self.Dv(self.phi, eps, self.alpha)) * self.ilk).real, 0.0]
        else:
            h = - ifftn(fftn(self.Dv(self.phi, eps, self.alpha)) * self.ilk).real
            sol = root(self.conjug_eq, h.flatten(), args=(eps, L), method=method, options={'fatol': 1e-9})
            if sol.success:
                return [sol.x.reshape((L, L)), 0.0]
            else:
                return [h, 0.0]

    def conjug_eq(self, h, eps, L):
        arg_v = (self.phi + 2.0 * xp.pi * xp.tensordot(self.alpha, h.reshape(self.dim * (L,)), axes=0)) % (2.0 * xp.pi)
        return (ifftn(self.lk * fftn(h.reshape(self.dim * (L,)))).real + self.Dv(arg_v, eps, self.alpha)).flatten()

    def refine_h(self, h, lam, eps):
        self.set_var(h.shape[0])
        fft_h = fftn(h)
        fft_h[self.zero_] = 0.0
        fft_h[xp.abs(fft_h) <= self.Threshold * xp.abs(fft_h).max()] = 0.0
        h_thresh = ifftn(fft_h).real
        arg_v = (self.phi + 2.0 * xp.pi * xp.tensordot(self.alpha, h_thresh, axes=0)) % (2.0 * xp.pi)
        fft_l = 1j * self.alpha_nu * fft_h
        fft_l[self.zero_] = self.rescale_fft
        l = ifftn(fft_l).real
        epsilon = ifftn(self.lk * fft_h).real + self.Dv(arg_v, eps, self.alpha) + lam
        fft_leps = fftn(l * epsilon)
        delta = - fft_leps[self.zero_].real / fft_l[self.zero_].real
        w = ifftn((delta * fft_l + fft_leps) * self.sml_div).real
        ll = l * ifftn(fft_l * self.exp_alpha_nu.conj()).real
        fft_wll = fftn(w / ll)
        fft_ill = fftn(1.0 / ll)
        w0 = - fft_wll[self.zero_].real / fft_ill[self.zero_].real
        beta = ifftn((fft_wll + w0 * fft_ill) * self.sml_div.conj()).real
        h_ = h_thresh + beta * l - xp.mean(beta * l) * l / xp.mean(l)
        lam_ = lam + delta
        fft_h_ = fftn(h_)
        fft_h_[self.zero_] = 0.0
        fft_h_[xp.abs(fft_h_) <= self.Threshold * xp.abs(fft_h_).max()] = 0.0
        tail_norm = xp.abs(fft_h_[self.tail_indx]).max()
        if self.AdaptSize and (tail_norm >= self.TolMin * xp.abs(fft_h_).max()) and (h.shape[0] < self.Lmax):
            self.set_var(2 * h.shape[0])
            h = ifftn(ifftshift(xp.pad(fftshift(fft_h), self.pad))).real * (2 ** self.dim)
            fft_h_ = ifftshift(xp.pad(fftshift(fft_h_), self.pad)) * (2 ** self.dim)
        h_ = ifftn(fft_h_).real
        arg_v = (self.phi + 2.0 * xp.pi * xp.tensordot(self.alpha, h_, axes=0)) % (2.0 * xp.pi)
        err = xp.abs(ifftn(self.lk * fft_h_).real + self.Dv(arg_v, eps, self.alpha) + lam_).max()
        if self.MonitorGrad:
            dh_ = self.id + 2.0 * xp.pi * xp.tensordot(self.alpha, xp.gradient(h_, 2.0 * xp.pi / h.shape[0]), axes=0)
            det_h_ = xp.abs(LA.det(xp.moveaxis(dh_, [0, 1], [-2, -1]))).min()
            if det_h_ <= self.TolMin:
                print('\033[31m        warning: non-invertibility...\033[00m')
        return h_, lam_, err

    def norms(self, h, r=0):
        self.set_var(h.shape[0])
        fft_h = fftn(h)
        if hasattr(self, 'alpha_perp'):
            return [xp.sqrt((xp.abs(ifftn(self.alpha_nu ** r * fft_h)) ** 2).sum()), xp.sqrt((xp.abs(ifftn(self.alpha_perp_nu ** r * fft_h)) ** 2).sum())]
        else:
            return xp.sqrt((xp.abs(ifftn(self.alpha_nu ** r * fft_h)) ** 2).sum())

if __name__ == "__main__":
    main()
