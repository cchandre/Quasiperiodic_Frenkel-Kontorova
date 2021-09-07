########################################################################################################################
##               Dictionary of parameters: https://github.com/cchandre/Quasiperiodic_Frenkel-Kontorova                ##
########################################################################################################################

import numpy as xp

Method = 'region'
#Method = 'line_norm'
Nxy = 512
r = 4

# omega = 0.618033988749895
# alpha = [1.0]
# Dv = lambda phi, eps, omega: - omega[0] / ((2.0 * xp.pi) ** 2) * (eps[0] * xp.sin(phi[0]) + eps[1] / 2.0 * xp.sin(2.0 * phi[0]))
# CoordRegion = [[0.0, 2.0], [-0.8, 0.8]]
# IndxLine = (0, 1)
# PolarAngles = [0.0, xp.pi / 2.0]
# CoordLine = [0.0, 0.028]
# ModesLine = (1, 1)
# DirLine = [1, 1]

omega = 1.0
alpha = [1.246979603717467, 2.801937735804838]
alpha_perp = [2.801937735804838, -1.246979603717467]
Dv = lambda phi, eps, omega: omega[0] / (2.0 * xp.pi) * eps[0] * xp.sin(phi[0]) + omega[1] / (2.0 * xp.pi) * eps[1] * xp.sin(phi[1])
#Dv = lambda phi, eps, omega: omega[0] / (2.0 * xp.pi) * (eps[0] * xp.sin(2.0 * phi[0] + 2.0 * phi[1]) + eps[1] * xp.sin(phi[0])) + omega[1] / (2.0 * xp.pi) * (eps[0] * xp.sin(2.0 * phi[0] + 2.0 * phi[1]) + eps[1] * xp.sin(phi[1]))
#CoordRegion = [[0.0, 0.02], [0.0,  0.004]]
CoordRegion = [[0.008, 0.014], [0.0025,  0.0035]]
IndxLine = (0, 1)
PolarAngles = [0.0, xp.pi / 2.0]
CoordLine = [0.0, 0.05]
ModesLine = (1, 1)
DirLine = [1, 5]

AdaptSize = False
Lmin = 2 ** 9
Lmax = 2 ** 9

TolMax = 1e10
TolMin = 1e-8
Threshold = 1e-10
MaxIter = 100

Type = 'cartesian'
ChoiceInitial = 'continuation'
MethodInitial = 'one_step'

AdaptEps = False
MinEps = 1e-6
MonitorGrad = False

Precision = 64
SaveData = False
PlotResults = True
Parallelization = (True, 4)

########################################################################################################################
##                                                DO NOT EDIT BELOW                                                   ##
########################################################################################################################
Precision = {32: xp.float32, 64: xp.float64, 128: xp.float128}.get(Precision, xp.float64)
dict = {'Method': 'compute_' + Method}
dict.update({
        'Nxy': Nxy,
        'r': r,
		'omega': omega,
		'alpha': xp.asarray(alpha, dtype=Precision),
		'Dv': Dv,
		'CoordRegion': xp.asarray(CoordRegion, dtype=Precision),
        'IndxLine': IndxLine,
        'PolarAngles': xp.asarray(PolarAngles, dtype=Precision),
		'CoordLine': xp.asarray(CoordLine, dtype=Precision),
		'ModesLine': xp.asarray(ModesLine),
		'DirLine': xp.asarray(DirLine),
		'AdaptSize': AdaptSize,
		'Lmin': Lmin,
		'Lmax': Lmax,
		'TolMax': TolMax,
		'TolMin': TolMin,
		'Threshold': Threshold,
		'MaxIter': MaxIter,
		'Type': Type,
		'ChoiceInitial': ChoiceInitial,
        'MethodInitial': MethodInitial,
		'AdaptEps': AdaptEps,
		'MinEps': MinEps,
		'MonitorGrad': MonitorGrad,
		'Precision': Precision,
		'SaveData': SaveData,
		'PlotResults': PlotResults,
		'Parallelization': Parallelization})
if 'alpha_perp' in locals():
    dict.update({'alpha_perp': xp.asarray(alpha_perp, dtype=Precision)})
########################################################################################################################
