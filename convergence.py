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
	h_ = h.copy()
	lam_ = lam.copy()
	if len(h_) == 0:
		h_, lam_ = case.initial_h([eps1, eps2])
	err = 1.0
	it_count = 0
	while case.tolmax >= err >= case.tolmin:
		h_, lam_, err = case.refine_h(h_, lam_, [eps1, eps2])
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
				point_ = lambda eps1: point(eps1, eps2, case)
				for result, h, lam in pool.imap(point_, iterable=eps_grid[0]):
					data.append(result)
			else:
				h = []
				lam = []
				for eps1 in tqdm(eps_grid[0], leave=False):
					result, h_, lam_ = point(eps1, eps2, case, h=h, lam=lam)
					if result[0] == 1:
						h = h_
						lam = lam_
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
					result, h_, lam_ = point(radius * xp.cos(theta), radius * xp.sin(theta), case, h=h, lam=lam)
					if result[0] == 1:
						h = h_
						lam = lam_
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
