import numpy as np
import math
import itertools
import sys
import matplotlib.pyplot as plt
import copy

from mpl_toolkits import mplot3d
from ripser import ripser
from persim import plot_diagrams
from scipy.spatial import distance_matrix
from scipy.spatial import distance

import cup_quasi_periodic_detection as cqpd

# Generate data set 

n = 2000

np.random.seed(13)

# Generate time series for sphere

# Base point
phi_0 = np.pi/3
theta_0 = np.pi/3

# time variable
t = np.linspace(0, 60*2*np.pi, 2000)

angles = np.column_stack((t, (1/60)*(1/2)*t))

def s2 (phi, theta):
    x = np.cos(phi)*np.sin(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(theta)
    
    return np.array([x,y,z])

points_0 = s2(phi_0, theta_0)

points_s2 = np.array([s2(l[0], l[1]) for l in angles])

ts_1 = np.arccos(np.dot(points_s2, points_0))

ts_1 = (1/(np.max(ts_1) - np.min(ts_1)))*ts_1

# Generate time series for circle 1

ts_2 = (1/3)*(np.cos(t) + (1/3)*np.cos(3*t) + (1/5)*np.cos(5*t))

# Generate time series for circle 2

ts_3 = (1/2)*(np.sin(t) + 0* np.sin(3*t) - (1/5)*np.sin(5*t))

# Concatenate time series to obtain one time series

ts = np.concatenate((ts_2 - ts_2[-1] + ts_1[0] , ts_1, ts_3 - ts_3[0] + ts_1[-1]))

t_1 = t
t_2 = t + np.max(t) + t[1]
t_3 = t + np.max(t_2) + t[1]

t_long = np.concatenate((t_1, t_2, t_3))

# Sliding window embedding

d = 6
w = 2*np.pi*d/(d+1)

pc = cqpd.sw_map(ts, t_long, w, d, 10000)

dm_X = distance.cdist(pc, pc)


# Maxmin subsampling

ind_L, cover_r = cqpd.maxmin(dm_X, 300)

dm_L = dm_X[ind_L,:][:,ind_L]

# Persistent homology

q = 2

result = ripser(dm_L, coeff=q, distance_matrix=True, maxdim=2, do_cocycles=True)

diagrams = result['dgms']
cocycles = result['cocycles']

# H_1 processing and sorting

H_1 = cocycles[1]
H_1_diagram = diagrams[1]
H_1_persistence = H_1_diagram[:,1] - H_1_diagram[:,0]
H_1_persistence_sort_ind = H_1_persistence.argsort()

# Cup product 

cocycle_1_ind = -1
cocycle_2_ind = -2

cocycle_1 = H_1[H_1_persistence_sort_ind[cocycle_1_ind]]
cocycle_2 = H_1[H_1_persistence_sort_ind[cocycle_2_ind]]

cup = cqpd.cup_product_cochains(cocycle_1, cocycle_2, q)

# Rips complex

rips_com, orders, diameters = cqpd.rips_complex(dm_L, 0.5)

# Coboundary matrices

d0 = cqpd.delta_0(rips_com['0'], rips_com['1'])
d1 = cqpd.delta_1(rips_com['1'], rips_com['2'])

# delta^1 reducition

R1, V1, low = cqpd.matrix_reduction(d1)

# Cup product as a cochain 

cochain = np.zeros(rips_com['2'].shape[0])

cup = np.array(cup)

for i in range(len(cup)):
    simplex = cup[i,2::-1]
    
    j = np.where((rips_com['2'] == simplex).all(axis=1))
    
    cochain[j] = cup[i,3]
    
cochain = cochain[::-1]

# Backwards substitution

y, index = cqpd.backwards_substitution(R1, low, cochain)

# Cohomological death
if index != -1:
	ind_goal = rips_com['2'][len(rips_com['2'])-index-1]
else:
	ind_goal = rips_com['2'][-1]

death = np.max(dm_L[np.ix_(ind_goal[::-1],ind_goal[::-1])])

# Cohomological birth: as minimum of birth between classes

birth = min(H_1_diagram[H_1_persistence_sort_ind[cocycle_1_ind]][1], H_1_diagram[H_1_persistence_sort_ind[cocycle_2_ind]][1])

# Plot of persistent diagram + cup product

plt.scatter(diagrams[0][:,0], diagrams[0][:,1], label='H_0')
plt.scatter(diagrams[1][:,0], diagrams[1][:,1], label='H_1')
plt.scatter(diagrams[2][:,0], diagrams[2][:,1], label='H_2')

plt.scatter([death], [birth], label ='Cup')

chosen = diagrams[1][[H_1_persistence_sort_ind[cocycle_1_ind], H_1_persistence_sort_ind[cocycle_2_ind]],:]

plt.scatter(chosen[:,0], chosen[:,1], label='Selected classes', facecolors='none', edgecolors='r')

plt.legend(loc = 'lower right')

pad = .03

x_min = np.min([np.min(diagrams[0][:,0]), np.min(diagrams[1][:,0]), np.min(diagrams[2][:,0])])
x_max = np.max([np.max(diagrams[0][:,0]), np.max(diagrams[1][:,0]), np.max(diagrams[2][:,0])])

y_min = np.min([np.min(diagrams[0][:,1]), np.min(diagrams[1][:,1]), np.min(diagrams[2][:,1])])
y_max = np.max([np.nanmax(diagrams[0][:,1][diagrams[0][:,1] != np.inf]), 
                np.nanmax(diagrams[1][:,1][diagrams[1][:,1] != np.inf]), 
                np.nanmax(diagrams[2][:,1][diagrams[2][:,1] != np.inf])])

plt.plot(np.linspace(x_min - pad, x_max + 2*pad), np.linspace(x_min - pad, x_max + 2*pad), '--', c='black')
plt.hlines(y_max + pad, x_min - pad, x_max + 2*pad, linestyles='--')

plt.xlim(x_min - pad, x_max + 2*pad)
plt.ylim(0 - pad, y_max + 2*pad)

plt.savefig('cup_persistence.png')