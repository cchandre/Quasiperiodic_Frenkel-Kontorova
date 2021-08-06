import numpy as xp
from tqdm import tqdm
import multiprocess
from scipy.io import savemat
import time
from datetime import date
import matplotlib.pyplot as plt


def save_data(name, data, timestr, case, info=[]):
    if case.save_results:
        mdic = case.DictParams.copy()
        mdic.update({'data': data, 'info': info})
        date_today = date.today().strftime(" %B %d, %Y\n")
        mdic.update({'date': date_today, 'author': 'cristel.chandre@univ-amu.fr'})
        savemat(type(case).__name__ + '_' + name + '_' + timestr + '.mat', mdic)


def point(eps, case, h, lam, gethull=False, display=False):
    h_ = h.copy()
    lam_ = lam
    err = 1.0
    it_count = 0
    while (case.tolmax >= err >= case.tolmin) and (it_count <= case.maxiter):
        h_, lam_, err = case.refine_h(h_, lam_, eps)
        it_count += 1
        if display:
            print(['...', it_count, err])
    if err <= case.tolmin:
        it_count = 0
    if gethull and case.save_results:
        timestr = time.strftime("%Y%m%d_%H%M")
        save_data('hull', h_, timestr, case)
    return [int(err <= case.tolmin), it_count], h_, lam_


def line_norm(case, display=True):
    timestr = time.strftime("%Y%m%d_%H%M")
    case.set_var(case.n_min)
    eps_modes = xp.array(case.eps_modes)
    eps_dir = xp.array(case.eps_dir)
    epsilon0 = case.eps_line[0]
    epsvec = epsilon0 * eps_modes * eps_dir + (1 - eps_modes) * eps_dir
    h, lam = case.initial_h(epsvec)
    deps0 = case.deps
    resultnorm = []
    count_fail = 0
    while epsilon0 <= case.eps_line[1] and (count_fail <= case.maxiter):
        deps = deps0
        epsilon = epsilon0 + deps
        epsvec = epsilon * eps_modes * eps_dir + (1 - eps_modes) * eps_dir
        if case.choice_initial == 'fixed':
            h, lam = case.initial_h(epsvec)
        result, h_, lam_ = point(epsvec, case, h, lam, display=False)
        if result[0] == 1:
            count_fail = 0
            resultnorm.append(xp.concatenate((epsilon, case.norms(h_, case.r)), axis=None))
            if display:
                print('For epsilon = {:.6f}, norm_{:d} = {:.3e}'.format(epsilon, case.r, case.norms(h_, case.r)[0]))
            if case.save_results:
                save_data('line_norm', xp.array(resultnorm), timestr, case)
        elif case.adapt_eps:
            while (result[0] == 0) and deps >= case.dist_surf:
                deps = deps / 10.0
                epsilon = epsilon0 + deps
                epsvec = epsilon * eps_modes * eps_dir + (1 - eps_modes) * eps_dir
                result, h_, lam_ = point(epsvec, case, h, lam, display=False)
            if result[0] == 1:
                count_fail = 0
                resultnorm.append(xp.concatenate((epsilon, case.norms(h_, case.r)), axis=None))
                if display:
                    print('For epsilon = {:.6f}, norm_{:d} = {:.3e}'.format(epsilon, case.r, case.norms(h_, case.r)[0]))
                if case.save_results:
                    save_data('line_norm', xp.array(resultnorm), timestr, case)
        if result[0] == 0:
            count_fail += 1
        if (result[0] == 1) and (case.choice_initial == 'continuation'):
            h = h_.copy()
            lam = lam_
        epsilon0 = epsilon
    resultnorm = xp.array(resultnorm)
    if case.plot_results:
        plt.semilogy(resultnorm[:, 0], resultnorm[:, 1], linewidth=2)
        plt.show()
    return resultnorm


def line(epsilon, case, getnorm=[False, 0], display=False):
    h, lam = case.initial_h(epsilon[0])
    results = []
    resultnorm = []
    for eps in tqdm(epsilon, disable=not display):
        result, h_, lam_ = point(eps, case, h, lam)
        if getnorm[0]:
            resultnorm.append(case.norms(h_, getnorm[1]))
        if (result[0] == 1) and case.choice_initial == 'continuation':
            h = h_.copy()
            lam = lam_
        elif case.choice_initial == 'fixed':
            h, lam = case.initial_h(eps)
        results.append(result)
    if getnorm[0]:
        if case.save_results:
            save_data('line_norm', xp.array(resultnorm), time.strftime("%Y%m%d_%H%M"), case, info=epsilon)
        if case.plot_results:
            plt.plot(xp.array(resultnorm))
            plt.show()
        return xp.array(resultnorm)
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
    convs = []
    iters = []
    if case.parallelization:
        num_cores = multiprocess.cpu_count()
        pool = multiprocess.Pool(num_cores)
        line_ = lambda it: line(eps_list[it], case)
        for conv, iter in tqdm(pool.imap(line_, iterable=range(case.eps_n)), total=case.eps_n):
            convs.append(conv)
            iters.append(iter)
    else:
        for it in trange(case.eps_n):
            conv, iter = line(eps_list[it], case)
            convs.append(conv)
            iters.append(iter)
    if case.save_results:
        save_data('region', xp.array(convs), timestr, case, info=xp.array(iters))
    if case.plot_results:
        if (case.eps_type == 'cartesian') and case.plot_results:
            plt.pcolor(xp.array(iters))
        elif (case.eps_type == 'polar') and case.plot_results:
            r, theta = xp.meshgrid(radii, thetas)
            fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
            ax.contourf(theta, r, xp.array(iters))
        plt.show()
    return xp.array(convs)
