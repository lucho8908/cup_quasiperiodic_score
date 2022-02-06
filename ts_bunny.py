import numpy as np
import math
import itertools
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
import os

from mpl_toolkits import mplot3d
from ripser import ripser
from persim import plot_diagrams
from scipy.spatial import distance_matrix
from scipy.spatial import distance
from matplotlib.ticker import StrMethodFormatter, FormatStrFormatter

import cup_quasi_periodic_detection as cqpd

# Create folder to store results

os.system('mkdir ts_bunny')

# Generate data set 

# t1 = np.linspace(0, 60*2*np.pi, 2000, endpoint=False)
t1 = np.linspace(0, 30*2*np.pi, 2000, endpoint=False)

curve1 = np.array([np.zeros(len(t1)), -np.sin(t1), 2 - np.cos(t1)]).transpose()

# t2 = np.linspace(60*2*np.pi, 120*2*np.pi, 2000, endpoint=False)
t2 = np.linspace(30*2*np.pi, 60*2*np.pi, 2000, endpoint=False)

angles = np.column_stack((t2, (1/30)*(1/2)*t2))

def s2 (phi, theta):
    x = np.cos(phi)*np.sin(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(theta)
    
    return np.array([x,y,z])

curve2 = np.array([s2(l[0], l[1]) for l in angles])
curve2 = curve2[::-1]

# t3 = np.linspace(120*2*np.pi, 180*2*np.pi, 2000, endpoint=False)
t3 = np.linspace(60*2*np.pi, 90*2*np.pi, 2000, endpoint=False)

curve3 = np.array([np.sin(t3), np.zeros(len(t3)), -2 + np.cos(t3)]).transpose()

complete_t = np.row_stack((t1.reshape([-1,1]), t2.reshape([-1,1]), t3.reshape([-1,1])))

complete_curve = np.row_stack([curve1, curve2, curve3])

time_series = np.arccos(np.dot(complete_curve, [1/4, 1/4, 1/4]))

# Plot time series
fig = plt.figure(figsize=(15,5))

ax = plt.plot(complete_t, time_series)
plt.xlabel('t')
plt.grid()
plt.box(on=None)

x_ticks = complete_t[0: len(complete_t) : int(len(complete_t)/7)]
x_ticks = x_ticks.reshape(-1)

plt.xticks(x_ticks, x_ticks)

plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.savefig('ts_bunny/time_series.png')
plt.close()

d = 5
w = 2*np.pi*d/(d+1)
tau = w/d

pc = cqpd.sw_map(time_series, complete_t, d, tau, 10000)

dm_X = distance.cdist(pc, pc)


# Maxmin subsampling

ind_L, cover_r = cqpd.maxmin(dm_X, 250)

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

print(cup)

# checking H2

# H_1 processing and sorting

H_2 = cocycles[2]
H_2_diagram = diagrams[2]
H_2_persistence = H_2_diagram[:,1] - H_2_diagram[:,0]
H_2_persistence_sort_ind = H_2_persistence.argsort()

print(H_2[H_2_persistence_sort_ind[-1]])

# Rips complex

temp = np.sort(H_1_diagram[:,1])
treshhold = temp[-2]

rips_com, orders, diameters = cqpd.rips_complex(dm_L, 0.8)

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

print(cochain)

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

fig = plt.figure(figsize=(10,10), dpi=96)

plt.scatter(diagrams[0][:,0], diagrams[0][:,1], label='H_0', s=80)
plt.scatter(diagrams[1][:,0], diagrams[1][:,1], label='H_1', s=80)
plt.scatter(diagrams[2][:,0], diagrams[2][:,1], label='H_2', s=80)

plt.savefig('ts_bunny/ubupersistence.png')

plt.scatter([death], [birth], label ='Cup', s=80)

chosen = diagrams[1][[H_1_persistence_sort_ind[cocycle_1_ind], H_1_persistence_sort_ind[cocycle_2_ind]],:]

plt.scatter(chosen[:,0], chosen[:,1], label='Selected classes', facecolors='none', edgecolors='r', s=(80 * 3))

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

plt.gca().set_aspect('equal')

plt.savefig('ts_bunny/cup_persistence.png', bbox_inches = 'tight')
plt.close()

# Plot of persistent diagram 

fig = plt.figure(figsize=(10,10), dpi=96)

plt.scatter(diagrams[0][:,0], diagrams[0][:,1], label='H_0', s=80)
plt.scatter(diagrams[1][:,0], diagrams[1][:,1], label='H_1', s=80)
plt.scatter(diagrams[2][:,0], diagrams[2][:,1], label='H_2', s=80)

# plt.savefig('ts_torus/persistence.png')

# plt.scatter([death], [birth], label ='Cup')

# chosen = diagrams[1][[H_1_persistence_sort_ind[cocycle_1_ind], H_1_persistence_sort_ind[cocycle_2_ind]],:]

# plt.scatter(chosen[:,0], chosen[:,1], label='Selected classes', facecolors='none', edgecolors='r', s=(mpl.rcParams['lines.s'] ** 3))

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

plt.gca().set_aspect('equal')

plt.savefig('ts_bunny/persistence.png', bbox_inches = 'tight')
plt.close()