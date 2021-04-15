import numpy as xp
from tqdm import tqdm
import multiprocess
from scipy.io import savemat
import time
import copy
from datetime import date
import matplotlib.pyplot as plt


def save_data(name, data, timestr, case, info=[]):
    if case.save_results:
        mdic = case.DictParams.copy()
        mdic.update({'data': data, 'info': info})
        date_today = date.today().strftime(" %B %d, %Y\n")
        mdic.update({'date': date_today, 'author': 'cristel.chandre@univ-amu.fr'})
        savemat(type(case).__name__ + '_' + name + '_' + timestr + '.mat', mdic)


def point(eps, case, hull, gethull=False, getnorm=[False, 0]):
    hull_ = copy.deepcopy(hull)
    err = 1.0
    it_count = 0
    while (case.tolmax >= err >= case.tolmin) and (it_count <= case.maxiter):
        hull_, err = case.image_h(hull_, eps)
        print(err)
        it_count += 1
    if err <= case.tolmin:
        it_count = 0
    if gethull:
        timestr = time.strftime("%Y%m%d_%H%M")
        save_data('hull', hull_.h, timestr, case)
    if getnorm[0]:
        return [int(err <= case.tolmin), it_count], hull_, case.norms(hull_, getnorm[1])
    return [int(err <= case.tolmin), it_count], hull_


def line(epsilon, case, getnorm=[False, 0], method=[], display=False):
    if method == 'critical':
        epsilon_ = xp.array(epsilon, dtype=case.precision)
        epsmin = epsilon_[case.eps_indx[0], 0]
        epsmax = epsilon_[case.eps_indx[0], 1]
        epsvec = epsilon_[:, 0].copy()
        while abs(epsmax - epsmin) >= case.dist_surf:
            epsmid = (epsmax + epsmin) / 2.0
            epsvec[case.eps_indx[0]] = epsmid * xp.cos(epsilon_[case.eps_indx[1], 0])
            epsvec[case.eps_indx[1]] = epsmid * xp.sin(epsilon_[case.eps_indx[1], 0])
            hull = case.initial_h(epsvec)
            if display:
                print([epsmin * xp.cos(epsilon_[case.eps_indx[1], 0]), epsmax * xp.cos(epsilon_[case.eps_indx[1], 0])])
            result, hull_ = point(epsvec, case, hull=hull)
            if result[0] == 1:
                epsmin = epsmid
            else:
                epsmax = epsmid
        return [epsmin * xp.cos(epsilon_[case.eps_indx[1], 0]), epsmin * xp.sin(epsilon_[case.eps_indx[1], 0])]
    else:
        hull = case.initial_h(epsilon[0])
        results = []
        for eps in epsilon:
            result, hull_ = point(eps, case, hull=hull)
            if result[0] == 1:
                hull = copy.deepcopy(hull_)
            else:
                hull = case.initial_h(eps)
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
    if (case.eps_type == 'cartesian') and case.plot_results:
        plt.pcolor(xp.array(convs))
    elif (case.eps_type == 'polar') and case.plot_results:
        r, theta = xp.meshgrid(radii, thetas)
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        ax.contourf(theta, r, xp.array(convs))
    plt.show()
    return xp.array(convs)
