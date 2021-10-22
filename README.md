# Analyticity breakdown for Frenkel-Kontorova models in quasiperiodic media

- [`qpfk_dict.py`](https://github.com/cchandre/Quasiperiodic_Frenkel-Kontorova/blob/main/qpfk_dict.py): to be edited to change the parameters of the qpFK computation (see below for a dictionary of parameters)

- [`qpfk.py`](https://github.com/cchandre/Quasiperiodic_Frenkel-Kontorova/blob/main/qpfk.py): contains the qpFK classes and main functions defining the qpFK map

- [`qpfk_modules.py`](https://github.com/cchandre/Quasiperiodic_Frenkel-Kontorova/blob/main/qpfk_modules.py): contains the methods to execute the qpFK map

Once [`qpfk_dict.py`](https://github.com/cchandre/Quasiperiodic_Frenkel-Kontorova/blob/main/qpfk_dict.py) has been edited with the relevant parameters, run the file as 
```sh
python3 qpfk.py
```

___
##  Parameter dictionary

- *Method*: 'line_norm', 'region'; choice of method                                            
- *Nxy*: integer; number of points along each line in computations 
- *r*: integer; order of the Sobolev norm used in `compute_line_norm()`                                        
####                                                                                                   
- *omega*: floats; frequency *&omega;*                                
- *alpha*: array of *n* floats; vector **&alpha;** defining the perturbation
- *alpha_perp* (optional): array of *n* floats; vector **&alpha;<sub>&perp;</sub>** perpendicular to **&alpha;**                             
- *Dv*: function; derivative of the *n*-d potential along a line                                               
- *CoordRegion*: array of floats; min and max values of the amplitudes for each mode of the potential (see *Dv*); used in `compute_region()`
- *IndxLine*: tuple of integers; indices of the modes to be varied in `compute_region()`                                        
         parallelization in `compute_region()` is done along the *IndxLine*[0] axis   
- *PolarAngles*: array of two floats; min and max value of the angles in 'polar'
- *CoordLine*: 1d array of floats; min and max values of the amplitudes of the potential used in `compute_line_norm()`   
- *ModesLine*: tuple of 0 and 1; specify which modes are being varied (1 for a varied mode)     
- *DirLine*: 1d array of floats; direction of the one-parameter family used in `compute_line_norm()`                 
####                                                                                           
                                         
####                                                                                                           
- *AdaptSize*: boolean; if True, changes the dimension of arrays depending on the tail of the FFT of *h*(*&psi;*)      
- *Lmin*: integer; minimum and default value of the dimension of arrays for *h*(*&psi;*)                           
- *Lmax*: integer; maximum value of the dimension of arrays for *h*(*&psi;*) if *AdaptSize* is True                   
####                                                                                                         
- *TolMax*: float; value of norm for divergence                                                      
- *TolMin*: float; value of norm for convergence                                                           
- *Threshold*: float; threshold value for truncating Fourier series of *h*(*&psi;*)                                   
- *MaxIter*: integer; maximum number of iterations for the Newton method                                      
####                                                                                                         
- *Type*: 'cartesian', 'polar'; type of computation for 2d plots                                             
- *ChoiceInitial*: 'fixed', 'continuation'; method for the initial conditions of the Newton method   
- *MethodInitial*: 'zero', 'one_step'; method to generate the initial conditions for the Newton iteration          
####                                                                                                       
- *AdaptEps*: boolean; if True adapt the increment of eps in `compute_line_norm()`                                   
- *MinEps*: float; minimum value of the increment of eps if *AdaptEps*=True                               
- *MonitorGrad*: boolean; if True, monitors the gradient of *h*(*&psi;*)                                      
####                                                                                 
- *Precision*: 32, 64 or 128; precision of calculations (default=64)                  
- *SaveData*: boolean; if True, the results are saved in a `.mat` file               
- *PlotResults*: boolean; if True, the results are plotted right after the computation              
- *Parallelization*: tuple (boolean, int); True for parallelization, int is the number of cores to be used (set int='all' for all of the cores)
####
---
For more information: <cristel.chandre@univ-amu.fr>


**References:**

1. R. Calleja, R. de la Llave, *Fast numerical computation of quasi-periodic equilibrium states in 1D statistical mechanics, including twist maps*, [Nonlinearity 22, 1311 (2009)](https://dx.doi.org/10.1088/0951-7715/22/6/004)
1. R. Calleja, R. de la Llave, *Computation of the breakdown of analyticity in statistical mechanics models: numerical results and a renormalization group explanation*, [Journal of Statistical Physics 141, 940 (2010)](https://dx.doi.org/10.1007/s10955-010-0085-7)
1. X. Su, R. de la Llave, *KAM theory for quasi-periodic equilibria in one-dimensional quasi-periodic media*, [SIAM Journal on Mathematical Analysis 44, 3901 (2012)](https://doi.org/10.1137/12087160X)
1. T. Blass, R. de la Llave, *The analyticity breakdown for Frenkel-Kontorova models in quasi-periodic media: numerical explorations*, [Journal of Statistical Physics 150, 1183 (2013)](https://dx.doi.org/10.1007/s10955-013-0718-8)


**Example: Figure 3(A) of Ref.[4]**

<img src="https://github.com/cchandre/Quasiperiodic_Frenkel-Kontorova/blob/main/qpFK_example.png" alt="Example" width="200"/>
