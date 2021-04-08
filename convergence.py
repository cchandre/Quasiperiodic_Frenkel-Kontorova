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
    while (case.tolmax >= err >= case.tolmin) and (it_count <= case.maxiter):
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
        else:
            h, lam = case.initial_h(eps)
        results.append(result)
    return results


def coord(case):
    eps_region = xp.array(case.eps_region)
    eps_vecs = xp.linspace(eps_region[:, 0], eps_region[:, 1], case.eps_n)
    if case.eps_type == 'cartesian':
        y = []
        for it in range(case.eps_n):
            eps_copy = eps_vecs.copy()
            eps_copy[:, case.indx[0]] = eps_vecs[it, case.indx[0]]
            y.append(eps_copy)
    elif case.eps_type == 'polar':
        thetas = eps_vecs[case.indx[1]]
        radii = eps_vecs[case.indx[0]]
        y = []
        for it in range(case.eps_n):
            eps_copy = eps_vecs.copy()
            eps_copy[:, case.indx[0]] = radii * xp.cos(thetas[it])
            eps_copy[:, case.indx[1]] = radii * xp.sin(thetas[it])
            y.append(eps_copy)
    return y


def region(case, scale='lin', output='all'):
    timestr = time.strftime("%Y%m%d_%H%M")
    num_cores = multiprocess.cpu_count()
    pool = multiprocess.Pool(num_cores)
    data = []
    eps_vecs = coord(case)
    line_ = lambda it: line(eps_vecs[it], case)
    for result in tqdm(pool.imap(line_, iterable=range(case.eps_n))):
        data.append(result)
    save_data('region', xp.array(data).reshape((case.eps_n, case.eps_n, -1)), timestr, case)
    return xp.array(data).reshape((case.eps_n, case.eps_n, -1))
