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
    if getnorm[0]:
        return [int(err <= case.tolmin), it_count], h_, lam_, case.norms(h_, getnorm[1])
    return [int(err <= case.tolmin), it_count], h_, lam_


def line(epsilon, case, getnorm=[False, 0], method=[], display=False):
    if method == 'critical':
        epsilon_ = xp.array(epsilon, dtype=case.precision)
        epsmin = epsilon_[case.eps_indx[0], 0]
        epsmax = epsilon_[case.eps_indx[0], 1]
        epsvec = epsilon_[:, 0].copy()
        h = []
        lam = []
        while abs(epsmax - epsmin) >= case.dist_surf:
            epsmid = (epsmax + epsmin) / 2.0
            epsvec[case.eps_indx[0]] = epsmid * xp.cos(epsilon_[case.eps_indx[1], 0])
            epsvec[case.eps_indx[1]] = epsmid * xp.sin(epsilon_[case.eps_indx[1], 0])
            if display:
                print([epsmin * xp.cos(epsilon_[case.eps_indx[1], 0]), epsmax * xp.cos(epsilon_[case.eps_indx[1], 0])])
            result, h_, lam_ = point(epsvec, case, h, lam)
            if result[0] == 1:
                epsmin = epsmid
                h = h_.copy()
                lam = lam_
            else:
                epsmax = epsmid
        return [epsmin * xp.cos(epsilon_[case.eps_indx[1], 0]), epsmin * xp.sin(epsilon_[case.eps_indx[1], 0])]
    else:
        h, lam = case.initial_h(epsilon[0])
        results = []
        for eps in epsilon:
            result, h_, lam_ = point(eps, case, h=h, lam=lam)
            if result[0] == 1:
                h = h_.copy()
                lam = lam_
            else:
                h = []
                lam = []
            results.append(result)
        return xp.array(results)[:, 0], xp.array(results)[:, 1]


def region(case):
    timestr = time.strftime("%Y%m%d_%H%M")
    eps_region = xp.array(case.eps_region, dtype=case.precision)
    eps_vecs = xp.linspace(eps_region[:, 0], eps_region[:, 1], case.eps_n, dtype=case.precision)
    if case.eps_type == 'cartesian':
        eps_list = []
        for it in range(case.eps_n):
            eps_copy = eps_vecs.copy()
            eps_copy[:, case.eps_indx[1]] = eps_vecs[it, case.eps_indx[1]]
            eps_list.append(eps_copy)
    elif case.eps_type == 'polar':
        thetas = eps_vecs[:, case.eps_indx[1]]
        radii = eps_vecs[:, case.eps_indx[0]]
        eps_list = []
        for it in range(case.eps_n):
            eps_copy = eps_vecs.copy()
            eps_copy[:, case.eps_indx[0]] = radii * xp.cos(thetas[it])
            eps_copy[:, case.eps_indx[1]] = radii * xp.sin(thetas[it])
            eps_list.append(eps_copy)
    num_cores = multiprocess.cpu_count()
    pool = multiprocess.Pool(num_cores)
    convs = []
    iters = []
    line_ = lambda it: line(eps_list[it], case)
    for conv, iter in tqdm(pool.imap(line_, iterable=range(case.eps_n)), total=case.eps_n):
        convs.append(conv)
        iters.append(iter)
    save_data('region', xp.array(convs), timestr, case, info=xp.array(iters))
    return xp.array(convs)
