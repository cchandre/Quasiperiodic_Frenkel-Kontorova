import numpy as xp
from numpy.fft import fftn, ifftn, fftfreq
import matplotlib.pyplot as plt
import warnings
import convergence as cv
warnings.filterwarnings("ignore")


def main():
    dict_params = {
    	'n': 2 ** 9,
    	'omega': 1.0,
    	'alpha': [1.246979603717467, 2.801937735804838],
    	'alpha_perp': [2.801937735804838, -1.246979603717467],
    	'potential': 'pot1_2d'}
    dict_params.update({
    	'eps_n': 30,
    	'eps_region': [[0.0, 0.05], [0.0,  0.001]],
    	'eps_type': 'cartesian'})
    # dict_params = {
    #     'n': 2 ** 10,
    #     'omega': 0.618033988749895,
    #     'alpha': [1.0],
    #     'potential': 'pot1_1d'}
    # dict_params.update({
    #     'eps_n': 128,
    #     'eps_region': [[0.0, 2.0], [-0.8, 0.8]],
    #     'eps_type': 'cartesian'})
    dict_params.update({
        'tolmax': 1e5,
        'tolmin': 1e-10,
        'threshold': 1e-14,
        'precision': 64,
        'save_results': False})
    alpha = dict_params['alpha']
    dv = {
        'pot1_1d': lambda phi, eps: - alpha[0] / (2.0 * xp.pi) * (eps[0] * xp.sin(phi[0]) + eps[1] / 2.0 * xp.sin(2.0 * phi[0])),
        'pot1_2d': lambda phi, eps: alpha[0] * eps[0] * xp.sin(phi[0]) + alpha[1] * eps[1] * xp.sin(phi[1]),
        'pot2_2d': lambda phi, eps: alpha[0] * (eps[0] * xp.sin(2.0 * phi[0] + 2.0 * phi[1]) + eps[1] * xp.sin(phi[0])) + alpha[1] * (eps[0] * xp.sin(2.0 * phi[0] + 2.0 * phi[1]) + eps[1] * xp.sin(phi[1]))
    }.get(dict_params['potential'], 'pot1_2d')
    case = qpFK(dv, dict_params)
    data = cv.region(case, parallel=False)
    plt.pcolor(data[:, :, 0])
    plt.show()


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
        dim = len(self.alpha)
        self.alpha = xp.array(self.alpha, self.precision)
        self.zero_ = dim * (0,)
        ind_nu = dim * (fftfreq(self.n, d=1.0/self.precision(self.n)),)
        nu = xp.asarray(xp.meshgrid(*ind_nu, indexing='ij'), dtype=self.precision)
        self.alpha_nu = xp.einsum('i,i...->...', self.alpha, nu)
        if hasattr(self, 'alpha_perp'):
            self.alpha_perp = xp.array(self.alpha_perp, self.precision)
            self.alpha_perp_nu = xp.einsum('i,i...->...', self.alpha_perp, nu)
        self.exp_alpha_nu = xp.exp(2j * xp.pi * self.omega * self.alpha_nu)
        self.lk_alpha_nu = 2.0 * (xp.cos(2.0 * xp.pi * self.omega * self.alpha_nu) - 1.0)
        self.sml_div = self.exp_alpha_nu - 1.0
        self.sml_div = xp.divide(1.0, self.sml_div, where=self.sml_div!=0)
        ind_phi = dim * (xp.linspace(0.0, 2.0 * xp.pi, self.n, endpoint=False),)
        self.phi = xp.asarray(xp.meshgrid(*ind_phi, indexing='ij'), dtype=self.precision)
        self.rescale_fft = self.n ** dim
        self.threshold *= self.rescale_fft
        self.ilk_alpha_nu = xp.divide(1.0, self.lk_alpha_nu, where=self.lk_alpha_nu!=0)
        self.initial_h = lambda eps: [- ifftn(fftn(self.dv(self.phi, eps)) * self.ilk_alpha_nu), 0.0]

    def refine_h(self, h, lam, eps):
        fft_h = fftn(h)
        fft_h[xp.abs(fft_h) <= self.threshold] = 0.0
        h_thresh = ifftn(fft_h)
        arg_v = self.phi + 2.0 * xp.pi * xp.tensordot(self.alpha, h_thresh, axes=0)
        lfunc = 1.0 + 2j * xp.pi * ifftn(self.alpha_nu * fft_h)
        epsilon = ifftn(self.lk_alpha_nu * fft_h) + lam + self.dv(arg_v, eps)
        fft_leps = fftn(lfunc * epsilon)
        fft_l = 2j * xp.pi * self.alpha_nu * fft_h * self.rescale_fft
        fft_l[self.zero_] = 1.0 * self.rescale_fft
        delta = - fft_leps[self.zero_] / fft_l[self.zero_]
        w = ifftn((delta * fft_l + fft_leps) * self.sml_div)
        ll = lfunc * ifftn(fft_l * self.exp_alpha_nu.conj())
        fft_wll = fftn(w / ll)
        fft_ill = fftn(1.0 / ll)
        w0 = - fft_wll[self.zero_] / fft_ill[self.zero_]
        delol = ifftn((fft_wll + w0 * fft_ill) * self.sml_div.conj())
        h = xp.real(h_thresh + delol * lfunc)
        lam = xp.real(lam + delta)
        arg_v = self.phi + 2.0 * xp.pi * xp.tensordot(self.alpha, h, axes=0)
        err = xp.abs(ifftn(self.lk_alpha_nu * fftn(h)) + lam + self.dv(arg_v, eps)).max()
        return h, lam, err

    def norms(self, h, r=0):
        if hasattr(self, 'alpha_perp'):
            return [xp.sqrt(xp.abs(ifftn(self.alpha_nu ** r * fftn(h)) ** 2).sum()), xp.sqrt(xp.abs(ifftn(self.alpha_perp_nu ** r * fftn(h)) ** 2).sum())]
        else:
            return xp.sqrt(xp.abs(ifftn(self.alpha_nu ** r * fftn(h)) ** 2).sum())


if __name__ == "__main__":
    main()
