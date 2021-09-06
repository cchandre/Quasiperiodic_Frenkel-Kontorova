import numpy as xp
from tqdm import tqdm, trange
import multiprocess
from scipy.io import savemat
import time
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.sans-serif': ['Palatino'],
    'font.size': 24,
    'axes.labelsize': 30,
    'figure.figsize': [8, 8],
    'image.cmap': 'bwr'})

def point(eps, h, lam, case, gethull=False, display=False):
    h_, lam_, err = h.copy(), lam, 1.0
    it_count = 0
    while (case.TolMax >= err >= case.TolMin) and (it_count <= case.MaxIter):
        h_, lam_, err = case.refine_h(h_, lam_, eps)
        if h_.shape[0] != h.shape[0]:
            h = case.pad_h(h)
        it_count += 1
        if display:
            print('\033[90m        iteration={:d}   err={:.3e} \033[00m'.format(it_count, err))
    if err <= case.TolMin:
        it_count = - it_count
    if gethull:
        timestr = time.strftime("%Y%m%d_%H%M")
        save_data('hull', h_, timestr, case)
    return [int(err <= case.TolMin), it_count], h_, lam_

def line(eps_list, case, display=False):
    h, lam = case.initial_h(eps_list[0], case.Lmin, case.MethodInitial)
    results = []
    for eps in tqdm(eps_list, disable=not display):
        result, h_, lam_ = point(eps, h, lam, case)
        if case.ChoiceInitial == 'continuation' and (result[0] == 1):
            h, lam = h_.copy(), lam_
        elif case.ChoiceInitial == 'fixed':
            h, lam = case.initial_h(eps, h_.shape[0], case.MethodInitial)
        results.append(result)
    return xp.array(results)[:, 0], xp.array(results)[:, 1]

def compute_line_norm(case, display=True):
    print('\033[92m    {} -- line_norm \033[00m'.format(case.__str__()))
    timestr = time.strftime("%Y%m%d_%H%M")
    epsilon0 = case.CoordLine[0]
    epsvec = epsilon0 * case.ModesLine * case.DirLine + (1 - case.ModesLine) * case.DirLine
    h, lam = case.initial_h(epsvec, case.Lmin, case.MethodInitial)
    deps = (case.CoordLine[1] - case.CoordLine[0]) / case.Precision(case.Nxy - 1)
    resultnorm, count_fail = [], 0
    while epsilon0 <= case.CoordLine[1] and (count_fail <= case.MaxIter):
        epsilon = epsilon0 + deps
        epsvec = epsilon * case.ModesLine * case.DirLine + (1 - case.ModesLine) * case.DirLine
        if case.ChoiceInitial == 'fixed':
            h, lam = case.initial_h(epsvec, h.shape[0], case.MethodInitial)
        result, h_, lam_ = point(epsvec, h, lam, case, display=False)
        if result[0] == 1:
            count_fail = 0
            resultnorm.append(xp.concatenate((epsilon, case.norms(h_, case.r)), axis=None))
            if display:
                print('\033[90m        epsilon={:.6f}    norm_{:d}={:.3e} \033[00m'.format(epsilon, case.r, case.norms(h_, case.r)[0]))
            save_data('line_norm', xp.array(resultnorm), timestr, case)
        elif case.AdaptEps:
            while (result[0] == 0) and deps >= case.MinEps:
                deps /= 5.0
                epsilon = epsilon0 + deps
                epsvec = epsilon * case.ModesLine * case.DirLine + (1 - case.ModesLine) * case.DirLine
                result, h_, lam_ = point(epsvec, h, lam, case, display=False)
            if result[0] == 1:
                count_fail = 0
                resultnorm.append(xp.concatenate((epsilon, case.norms(h_, case.r)), axis=None))
                if display:
                    print('\033[90m        epsilon={:.6f}    norm_{:d}={:.3e} \033[00m'.format(epsilon, case.r, case.norms(h_, case.r)[0]))
                save_data('line_norm', xp.array(resultnorm), timestr, case)
        if result[0] == 0:
            count_fail += 1
        elif (case.ChoiceInitial == 'continuation'):
            h, lam = h_.copy(), lam_
        epsilon0 = epsilon
    resultnorm = xp.array(resultnorm)
    if case.PlotResults and resultnorm.size != 0:
        fig, ax = plt.subplots(1, 1)
        ax.semilogy(resultnorm[:, 0], resultnorm[:, 1], linewidth=2)
        ax.set_xlabel('$\epsilon$')
        ax.set_ylabel('$\Vert h \Vert_{}$'.format(case.r))
    return resultnorm

def compute_region(case):
    print('\033[92m    {} -- region \033[00m'.format(case.__str__()))
    timestr = time.strftime("%Y%m%d_%H%M")
    eps_vecs = xp.linspace(case.CoordRegion[:, 0], case.CoordRegion[:, 1], case.Nxy, dtype=case.Precision)
    if case.Type == 'cartesian':
        eps_list = []
        for _ in range(case.Nxy):
            eps_ = eps_vecs.copy()
            eps_[:, case.IndxLine[1]] = eps_vecs[_, case.IndxLine[1]]
            eps_list.append(eps_)
    elif case.Type == 'polar':
        thetas = xp.linspace(case.PolarAngles[0], case.PolarAngles[1], case.Nxy, dtype=case.Precision)
        radii = xp.linspace(0.0, 1.0, case.Nxy, dtype=case.Precision)
        eps_list = []
        for _ in range(case.Nxy):
            eps_ = eps_vecs.copy()
            eps_[:, case.IndxLine[0]] = radii * xp.cos(thetas[_]) * case.CoordRegion[case.IndxLine[0], 1]
            eps_[:, case.IndxLine[1]] = radii * xp.sin(thetas[_]) * case.CoordRegion[case.IndxLine[1], 1]
            eps_list.append(eps_)
    convs, iters = [], []
    if case.Parallelization[0]:
        if case.Parallelization[1] == 'all':
            num_cores = multiprocess.cpu_count()
        else:
            num_cores = min(multiprocess.cpu_count(), case.Parallelization[1])
        pool = multiprocess.Pool(num_cores)
        line_ = lambda _: line(eps_list[_], case)
        for conv, iter in tqdm(pool.imap(line_, iterable=range(case.Nxy)), total=case.Nxy):
            convs.append(conv)
            iters.append(iter)
    else:
        for _ in trange(case.Nxy):
            conv, iter = line(eps_list[_], case)
            convs.append(conv)
            iters.append(iter)
    save_data('region', xp.array(convs), timestr, case, info=xp.array(iters))
    if case.PlotResults:
        divnorm = colors.TwoSlopeNorm(vmin=xp.amin(xp.array(iters)), vcenter=0.0, vmax=xp.amax(xp.array(iters)))
        if (case.Type == 'cartesian'):
            fig, ax = plt.subplots(1, 1)
            ax.set_box_aspect(1)
            im = ax.pcolormesh(eps_vecs[:, 0], eps_vecs[:, 1], xp.array(iters), norm=divnorm)
            ax.set_xlabel('$\epsilon_1$')
            ax.set_ylabel('$\epsilon_2$')
            fig.colorbar(im)
        elif (case.Type == 'polar'):
            r, theta = xp.meshgrid(radii, thetas)
            fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
            im = ax.contourf(theta, r, xp.array(iters), norm=divnorm)
            fig.colorbar(im)
    return xp.array(convs)

def save_data(name, data, timestr, case, info=[]):
    if case.SaveData:
        mdic = case.DictParams.copy()
        del mdic['Precision']
        mdic.update({'data': data, 'info': info})
        date_today = date.today().strftime(" %B %d, %Y")
        mdic.update({'date': date_today, 'author': 'cristel.chandre@univ-amu.fr'})
        name_file = type(case).__name__ + '_' + name + '_' + timestr + '.mat'
        savemat(name_file, mdic)
        print('\033[90m        Results saved in {} \033[00m'.format(name_file))
