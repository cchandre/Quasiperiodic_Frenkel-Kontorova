import numpy as xp
from numpy.fft import fftn, ifftn, fftfreq
import convergence as cv
import warnings
warnings.filterwarnings("ignore")


def main():
    dict_params = {
    	'n': 2 ** 9,
    	'omega': 1.0,
    	'alpha': [1.246979603717467, 2.801937735804838],
    	'alpha_perp': [2.801937735804838, -1.246979603717467],
    	'potential': 'pot1_2d'}
    dict_params.update({
    	'eps_n': 512,
    	'eps_region': [[0.0, 0.02], [0.0,  0.004]],
        'eps_indx': [0, 1],
    	'eps_type': 'cartesian'})
    # dict_params = {
    #     'n': 2 ** 10,
    #     'omega': 0.618033988749895,
    #     'alpha': [1.0],
    #     'potential': 'pot1_1d'}
    # dict_params.update({
    #     'eps_n': 512,
    #     'eps_region': [[0.0, 2.0], [0.0, 0.8]],
    #     'eps_indx': [0, 1],
    #     'eps_type': 'cartesian'})
    dict_params.update({
        'tolmax': 1e5,
        'tolmin': 1e-8,
        'dist_surf': 1e-5,
        'maxiter': 100,
        'threshold': 1e-9,
        'choice_initial': 'fixed',
        'precision': 64,
        'parallelization': [True, 8],
        'save_results': True,
        'plot_results': True})
    dv = {
        'pot1_1d': lambda phi, eps, alpha: - alpha[0] / (2.0 * xp.pi) * (eps[0] * xp.sin(phi[0]) + eps[1] / 2.0 * xp.sin(2.0 * phi[0])),
        'pot1_2d': lambda phi, eps, alpha: alpha[0] * eps[0] * xp.sin(phi[0]) + alpha[1] * eps[1] * xp.sin(phi[1]),
        'pot2_2d': lambda phi, eps, alpha: alpha[0] * (eps[0] * xp.sin(2.0 * phi[0] + 2.0 * phi[1]) + eps[1] * xp.sin(phi[0])) + alpha[1] * (eps[0] * xp.sin(2.0 * phi[0] + 2.0 * phi[1]) + eps[1] * xp.sin(phi[1]))
    }.get(dict_params['potential'], 'pot1_2d')
    case = qpFK(dv, dict_params)
    data = cv.region(case)


class qpFK:
    def __repr__(self):
        return '{self.__class__.__name__}({self.dv, self.DictParams})'.format(self=self)

    def __str__(self):
        return 'Quasiperiodic Frenkel-Kontorova ({self.__class__.__name__}) model with alpha = {self.alpha} and omega = {self.omega}'.format(self=self)

    def __init__(self, dv, dict_params):
        for key in dict_params:
            setattr(self, key, dict_params[key])
        self.DictParams = dict_params
        self.precision = {64: xp.float64, 128: xp.float128}.get(self.precision, xp.float64)
        self.dv = dv
        dim = len(self.alpha)
        self.alpha = xp.array(self.alpha, dtype=self.precision)
        self.zero_ = dim * (0,)
        ind_nu = dim * (fftfreq(self.n, d=1.0/self.precision(self.n)),)
        nu = xp.meshgrid(*ind_nu, indexing='ij')
        self.alpha_nu = 2.0 * xp.pi * xp.einsum('i,i...->...', self.alpha, nu)
        if hasattr(self, 'alpha_perp'):
            self.alpha_perp = xp.array(self.alpha_perp, dtype=self.precision)
            self.alpha_perp_nu = xp.einsum('i,i...->...', self.alpha_perp, nu)
        self.exp_alpha_nu = xp.exp(1j * self.omega * self.alpha_nu)
        self.lk = 2.0 * (xp.cos(self.omega * self.alpha_nu) - 1.0)
        self.sml_div = self.exp_alpha_nu - 1.0
        self.sml_div = xp.divide(1.0, self.sml_div, where=self.sml_div!=0)
        ind_phi = dim * (xp.linspace(0.0, 2.0 * xp.pi, self.n, endpoint=False, dtype=self.precision),)
        self.phi = xp.meshgrid(*ind_phi, indexing='ij')
        self.rescale_fft = self.precision(self.n ** dim)
        self.threshold *= self.rescale_fft
        ilk = xp.divide(1.0, self.lk, where=self.lk!=0)
        self.initial_h = lambda eps: [- ifftn(fftn(self.dv(self.phi, eps, self.alpha)) * ilk).real, 0.0]

    def refine_h(self, h, lam, eps):
        fft_h = fftn(h)
        fft_h[xp.abs(fft_h) <= self.threshold] = 0.0
        fft_h[self.zero_] = 0.0
        h_thresh = ifftn(fft_h).real
        arg_v = self.phi + 2.0 * xp.pi * xp.tensordot(self.alpha, h_thresh, axes=0)
        fft_l = 1j * self.alpha_nu * fft_h
        fft_l[self.zero_] = self.rescale_fft
        lfunc = ifftn(fft_l).real
        epsilon = ifftn(self.lk * fft_h).real + self.dv(arg_v, eps, self.alpha) + lam
        fft_leps = fftn(lfunc * epsilon)
        delta = - fft_leps[self.zero_].real / fft_l[self.zero_].real
        w = ifftn((delta * fft_l + fft_leps) * self.sml_div).real
        ll = lfunc * ifftn(fft_l * self.exp_alpha_nu.conj()).real
        fft_wll = fftn(w / ll)
        fft_ill = fftn(1.0 / ll)
        w0 = - fft_wll[self.zero_].real / fft_ill[self.zero_].real
        beta = ifftn((fft_wll + w0 * fft_ill) * self.sml_div.conj()).real
        h_ = h_thresh + beta * lfunc - xp.mean(beta * lfunc) * lfunc
        lam_ = lam + delta
        arg_v = self.phi + 2.0 * xp.pi * xp.tensordot(self.alpha, h_, axes=0)
        err = xp.abs(ifftn(self.lk * fftn(h_)).real + lam_ + self.dv(arg_v, eps, self.alpha)).max()
        return h_, lam_, err

    def norms(self, h, r=0):
        if hasattr(self, 'alpha_perp'):
            return [xp.sqrt(xp.abs(ifftn(self.alpha_nu ** r * fftn(h)) ** 2).sum()), xp.sqrt(xp.abs(ifftn(self.alpha_perp_nu ** r * fftn(h)) ** 2).sum())]
        else:
            return xp.sqrt(xp.abs(ifftn(self.alpha_nu ** r * fftn(h)) ** 2).sum())


if __name__ == "__main__":
    main()
