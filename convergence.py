import numpy as xp
from tqdm import tqdm
import multiprocess
from scipy.io import savemat
import time
from datetime import date


def save_data(name, data, timestr, case, info=[]):
    if case.save_results:
        mdic = case.DictParams.copy()
        mdic.update({'data': data, 'info': info})
        date_today = date.today().strftime(" %B %d, %Y\n")
        mdic.update({'date': date_today, 'author': 'cristel.chandre@univ-amu.fr'})
        savemat(type(case).__name__ + '_' + name + '_' + timestr + '.mat', mdic)


def point(eps, case, h=[], lam=[], gethull=False, getnorm=[False, 0]):
    h_ = h.copy()
    lam_ = lam
    if len(h_) == 0:
        h_, lam_ = case.initial_h(eps)
    err = 1.0
    it_count = 0
    while case.tolmax >= err >= case.tolmin:
        h_, lam_, err = case.refine_h(h_, lam_, eps)
        it_count += 1
    if err <= case.tolmin:
        it_count = 0
    if gethull:
        timestr = time.strftime("%Y%m%d_%H%M")
        save_data('hull', h_, timestr, case)
        return [int(err <= case.tolmin), it_count], h_, lam_
    if getnorm[0]:
        return [int(err <= case.tolmin), it_count], h_, lam_, case.norms(h_, getnorm[1])
    return [int(err <= case.tolmin), it_count], h_, lam_


def line(epsilon, case, getnorm=[False, 0]):
    h, lam = case.initial_h(epsilon[0])
    results = []
    for eps in epsilon:
        result, h_, lam_ = point(eps, case, h=h, lam=lam)
        if result[0] == 1:
            h = h_.copy()
            lam = lam_
        results.append(result)
    return results


def region(case, scale='lin', output='all'):
    timestr = time.strftime("%Y%m%d_%H%M")
    num_cores = multiprocess.cpu_count()
    pool = multiprocess.Pool(num_cores)
    eps_region = xp.array(case.eps_region)
    data = []
    if case.eps_type == 'cartesian':
        eps_perp = xp.linspace(eps_region[0, 0], eps_region[0, 1], case.eps_n)
        eps_para = xp.linspace(eps_region[1, 0], eps_region[1, 1], case.eps_n)
        line_ = lambda eps1: line(xp.array([eps1 * xp.ones_like(eps_para), eps_para]).transpose().reshape(-1, 2), case)
        for result in tqdm(pool.imap(line_, iterable=eps_perp)):
            data.append(result)
    elif case.eps_type == 'polar':
        thetas = xp.linspace(eps_region[1, 0], eps_region[1, 1], case.eps_n)
        if output == 'all':
            if scale == 'lin':
                radii = xp.linspace(eps_region[0, 0], eps_region[0, 1], case.eps_n)
            elif scale == 'log':
                radii = eps_region[0, 1] * (1.0 - xp.logspace(xp.log10((eps_region[0, 1] - eps_region[0, 0]) / eps_region[0, 1]), xp.log10(case.tolmin), case.eps_n))
            line_ = lambda theta: line(xp.array([radii * xp.cos(theta), radii * xp.sin(theta)]).transpose().reshape(-1, 2), case)
            for result in tqdm(pool.imap(line_, iterable=thetas)):
                data.append(result)
        elif output == 'critical':
            for theta in thetas:
                eps_min, eps_max = eps_region[0]
                while xp.abs(eps_min - eps_max) >= case.tolmin:
                    eps_mid = (eps_min + eps_max) / 2.0
                    if point([eps_mid * xp.cos(theta), eps_mid * xp.sin(theta)], case)[0]:
                        eps_min = eps_mid
                    else:
                        eps_max = eps_mid
                data.append(eps_min)
            return data
    save_data('region', xp.array(data).reshape((case.eps_n, case.eps_n, -1)), timestr, case)
    return xp.array(data).reshape((case.eps_n, case.eps_n, -1))
