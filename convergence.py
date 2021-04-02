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

def point(eps1, eps2, case, h = [], lam=[], gethull=False, getnorm=[False, 0]):
	if len(h) == 0:
		h, lam = case.initial_h([eps1, eps2])
	err = 1.0
	it_count = 0
	while case.tolmax >= err >= case.tolmin:
		h, lam, err = case.refine_h(h, lam, [eps1, eps2])
		it_count += 1
	if err <= case.tolmin:
		it_count = 0
	if gethull:
		timestr = time.strftime("%Y%m%d_%H%M")
		save_data('hull', h, timestr, case)
		return [int(err <= case.tolmin), it_count], h
	if getnorm[0]:
		return [int(err <= case.tolmin), it_count], h, case.norms(h, getnorm[1])
	return [int(err <= case.tolmin), it_count], h, lam

def region(case, scale='lin', output= 'all', parallel=False):
	timestr = time.strftime("%Y%m%d_%H%M")
	if parallel:
		num_cores = multiprocess.cpu_count()
		pool = multiprocess.Pool(num_cores)
	eps_region = xp.array(case.eps_region)
	data = []
	if case.eps_type == 'cartesian':
		eps_grid = [xp.linspace(eps_region[0, 0], eps_region[0, 1], case.eps_n), xp.linspace(eps_region[1, 0], eps_region[1, 1], case.eps_n)]
		for eps2 in tqdm(eps_grid[1]):
			if parallel:
				for result, h in pool.imap(lambda eps1: point(eps1, eps2, case), iterable=eps_grid[0]):
					data.append(result)
			else:
				for eps1 in tqdm(eps_grid[0], leave=False):
					result, h, lam = point(eps1, eps2, case)
					data.append(result)
			save_data('region', data, timestr, case)
	elif case.eps_type == 'polar':
		thetas = xp.linspace(eps_region[1, 0], eps_region[1, 1], case.eps_n)
		if output == 'all':
			if scale == 'lin':
				radii = xp.linspace(eps_region[0, 0], eps_region[0, 1], case.eps_n)
			elif scale == 'log':
				radii = eps_region[0, 1] * (1.0 - xp.logspace(xp.log10((eps_region[0, 1] - eps_region[0, 0]) / eps_region[0, 1]), xp.log10(case.tolmin), case.eps_n))
			for theta in tqdm(thetas):
				h = []
				lam = []
				for radius in tqdm(radii, leave=False):
					result, h, lam = point(radius * xp.cos(theta), radius * xp.sin(theta), case, h, lam)
					data.append(result)
				save_data('region', data, timestr, case)
		elif output == 'critical':
			for theta in thetas:
				eps_min, eps_max = eps_region[0]
				while xp.abs(eps_min - eps_max) >= case.tolmin:
					eps_mid = (eps_min + eps_max) / 2.0
					if point(eps_mid * xp.cos(theta), eps_mid * xp.sin(theta), case)[0]:
						eps_min = eps_mid
					else:
						eps_max = eps_mid
				data.append(eps_min)
	save_data('region', data, timestr, case)
	return xp.array(data).reshape((case.eps_n, case.eps_n, -1))
