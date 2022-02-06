import numpy as np
import math
import itertools
import sys
import matplotlib.pyplot as plt
import copy
import os

from mpl_toolkits import mplot3d
from ripser import ripser
from persim import plot_diagrams
from scipy.spatial import distance_matrix
from scipy.spatial import distance
from matplotlib.ticker import StrMethodFormatter, FormatStrFormatter
from sklearn import metrics
from sklearn.preprocessing import label_binarize

import cup_quasi_periodic_detection as cqpd

# Create folder to store results

os.system('rm -r roc_curves/*')
os.system('rmdir roc_curves')

os.system('mkdir roc_curves')

# np.random.seed(10)

def f(t, A=1, B=1, f1=1, f2=np.sqrt(3)):
    return A*np.sin(f1*t) + B*np.sin(f2*t)

def linear_f(t, b=0, m=1):
	return m*t + b

def f_bunny(point):

	t1 = np.linspace(0, 60*2*np.pi, 2000, endpoint=False)

	curve1 = np.array([np.zeros(len(t1)), -np.sin(t1), 2 - np.cos(t1)]).transpose()

	t2 = np.linspace(60*2*np.pi, 120*2*np.pi, 2000, endpoint=False)

	angles = np.column_stack((t2, (1/60)*(1/2)*t2))

	def s2 (phi, theta):
	    x = np.cos(phi)*np.sin(theta)
	    y = np.sin(phi)*np.sin(theta)
	    z = np.cos(theta)
	    
	    return np.array([x,y,z])

	curve2 = np.array([s2(l[0], l[1]) for l in angles])

	t3 = np.linspace(120*2*np.pi, 180*2*np.pi, 2000, endpoint=False)

	curve3 = np.array([np.sin(t3), np.zeros(len(t3)), -2 + np.cos(t3)]).transpose()

	complete_t = np.row_stack((t1.reshape([-1,1]), t2.reshape([-1,1]), t3.reshape([-1,1])))

	complete_curve = np.row_stack([curve1, curve2, curve3])

	# time_series = np.arccos(np.dot(complete_curve, point))

	time_series = np.power(distance.cdist(complete_curve, [point]).reshape(-1), 2)

	time_series = time_series - np.mean(time_series)

	return complete_t, time_series

# -----------------------------------------------------------------------------
# Global parameters
# -----------------------------------------------------------------------------
N = 5000

t = np.linspace(0, 200, N)

cup = []
first_h2 = []
second_h1 = []
real_quasi = []

# -----------------------------------------------------------------------------
# Bunny
# -----------------------------------------------------------------------------
print('Bunny')
for i in range(2):

	rng = np.random.default_rng()

	x = .25*rng.random()
	y = .25*rng.random()
	z = .25*rng.random()

	point = [1/4+x, 1/4+y, 1/4+z]

	t, time_series = f_bunny(point)

	plt.plot(t, time_series)
	plt.title(r'$point = ({}, {}, {}))$'.format(x,y,z))
	plt.savefig('roc_curves/bunny_plot_{}.png'.format(i))
	plt.close()

	tau, d = cqpd.optimal_tau_frequencies(time_series, t, np.sort([-1,0,1]))

	print(tau, d)

	# pc = cqpd.SW(f, t, d, tau)

	pc = cqpd.sw_map(time_series, t, d, tau, 1000)

	dm_X = distance.cdist(pc, pc)

	# Maxmin subsampling

	ind_L, cover_r = cqpd.maxmin(dm_X, 170)

	dm_L = dm_X[ind_L,:][:,ind_L]

	# Persistent homology

	q = 2

	result = ripser(dm_L, coeff=q, distance_matrix=True, maxdim=2, do_cocycles=True)

	# result = ripser(pc, coeff=q, maxdim=2, do_cocycles=True, n_perm=300)


	diagrams = result['dgms']
	cocycles = result['cocycles']

	plt.close()
	plot_diagrams(diagrams)
	plt.savefig('roc_curves/bunny_persistence_{}.png'.format(i))
	plt.close()

	co1_ind=-1
	co2_ind=-2

	cup_bar = cqpd.getQuasiPeriodicScore(diagrams, cocycles, dm_L, coeff=q, cocycle_1_ind=co1_ind, cocycle_2_ind=co2_ind)

	cup.append(cup_bar)
	real_quasi.append(2)

	# Plot new diagrams

	# Plot of persistent diagram + cup product
	plt.close()
	plt.scatter(diagrams[0][:,0], diagrams[0][:,1], label='H_0')
	plt.scatter(diagrams[1][:,0], diagrams[1][:,1], label='H_1')
	plt.scatter(diagrams[2][:,0], diagrams[2][:,1], label='H_2')

	plt.scatter(cup_bar[0], cup_bar[1], label ='Cup')

	H_1 = cocycles[1]
	H_1_diagram = diagrams[1]
	H_1_persistence = H_1_diagram[:,1] - H_1_diagram[:,0]
	H_1_persistence_sort_ind = H_1_persistence.argsort()

	second_h1.append(H_1_diagram[H_1_persistence_sort_ind[-2],:])

	chosen = diagrams[1][[H_1_persistence_sort_ind[co1_ind], H_1_persistence_sort_ind[co2_ind]],:]

	plt.scatter(chosen[:,0], chosen[:,1], label='Selected classes', facecolors='none', edgecolors='r')

	plt.legend(loc = 'lower right')

	plt.savefig('roc_curves/bunny_cup_persistence_{}.png'.format(i))
	plt.close()

	H_2 = cocycles[2]
	H_2_diagram = diagrams[2]
	H_2_persistence = H_2_diagram[:,1] - H_2_diagram[:,0]
	H_2_persistence_sort_ind = H_2_persistence.argsort()

	if H_2_diagram.shape[0] < 1:
		first_h2.append(np.array([0,0]))
	else:
		first_h2.append(H_2_diagram[H_2_persistence_sort_ind[-1],:])

	print(i)

t = np.linspace(0, 200, N)
# -----------------------------------------------------------------------------
# Linear
# -----------------------------------------------------------------------------
print('Linear')
for i in range(10):

	rng = np.random.default_rng()

	A = rng.integers(low=1, high=5)
	B = rng.integers(low=1, high=5)

	time_series = linear_f(t, A, B) + np.random.normal(0, 1, N)

	plt.plot(t, time_series)
	plt.title(r'$f(t) = {}t + {}$'.format(A,B))
	plt.savefig('roc_curves/linear_plot_{}.png'.format(i))
	plt.close()

	# tau, d = cqpd.optimal_tau_frequencies(time_series, t, np.sort([f1,-f1,f2,-f2]))

	tau, d = cqpd.optimal_tau2(time_series, t)

	pc = cqpd.SW(f, t, d, tau)

	# pc = cqpd.sw_map(time_series, t, d, tau, 5000)

	dm_X = distance.cdist(pc, pc)

	# Maxmin subsampling

	ind_L, cover_r = cqpd.maxmin(dm_X, 200)

	dm_L = dm_X[ind_L,:][:,ind_L]

	# Persistent homology

	q = 2

	result = ripser(dm_L, coeff=q, distance_matrix=True, maxdim=2, do_cocycles=True)

	# result = ripser(pc, coeff=q, maxdim=2, do_cocycles=True, n_perm=300)


	diagrams = result['dgms']
	cocycles = result['cocycles']

	plt.close()
	plot_diagrams(diagrams)
	plt.savefig('roc_curves/linear_persistence_{}.png'.format(i))
	plt.close()

	co1_ind=-1
	co2_ind=-2

	cup_bar = cqpd.getQuasiPeriodicScore(diagrams, cocycles, dm_L, coeff=q, cocycle_1_ind=co1_ind, cocycle_2_ind=co2_ind)

	cup.append(cup_bar)
	real_quasi.append(3)

	# Plot new diagrams

	# Plot of persistent diagram + cup product
	plt.close()
	plt.scatter(diagrams[0][:,0], diagrams[0][:,1], label='H_0')
	plt.scatter(diagrams[1][:,0], diagrams[1][:,1], label='H_1')
	plt.scatter(diagrams[2][:,0], diagrams[2][:,1], label='H_2')

	plt.scatter(cup_bar[0], cup_bar[1], label ='Cup')

	H_1 = cocycles[1]
	H_1_diagram = diagrams[1]
	H_1_persistence = H_1_diagram[:,1] - H_1_diagram[:,0]
	H_1_persistence_sort_ind = H_1_persistence.argsort()

	if H_1_diagram.shape[0] < 1:
		second_h1.append(np.array([0,0]))

		chosen = np.array([[0,0], [0,0]])
	else:
		second_h1.append(H_1_diagram[H_1_persistence_sort_ind[-2],:])

		chosen = diagrams[1][[H_1_persistence_sort_ind[co1_ind], H_1_persistence_sort_ind[co2_ind]],:]

	plt.scatter(chosen[:,0], chosen[:,1], label='Selected classes', facecolors='none', edgecolors='r')

	plt.legend(loc = 'lower right')

	plt.savefig('roc_curves/linear_cup_persistence_{}.png'.format(i))
	plt.close()

	H_2 = cocycles[2]
	H_2_diagram = diagrams[2]
	H_2_persistence = H_2_diagram[:,1] - H_2_diagram[:,0]
	H_2_persistence_sort_ind = H_2_persistence.argsort()

	if H_2_diagram.shape[0] < 1:
		first_h2.append(np.array([0,0]))
	else:
		first_h2.append(H_2_diagram[H_2_persistence_sort_ind[-1],:])

	print(i)

# -----------------------------------------------------------------------------
# Quasi-periodic
# -----------------------------------------------------------------------------
print('Quasi')
for i in range(10):

	rng = np.random.default_rng()

	A = rng.integers(low=1, high=5)
	B = rng.integers(low=1, high=5)
	f1 = rng.integers(low=1, high=5)
	f2 = np.sqrt(3)

	time_series = f(t, A, B, f1, f2)

	plt.plot(t, time_series)
	plt.title(r'$f(t) = {}sin({}t) + {}cos({}t)$'.format(A,B,f1,f2))
	plt.savefig('roc_curves/quasi_plot_{}.png'.format(i))
	plt.close()

	tau, d = cqpd.optimal_tau_frequencies(time_series, t, np.sort([f1,-f1,f2,-f2]))

	pc = cqpd.SW(f, t, d, tau)

	# pc = cqpd.sw_map(time_series, t, d, tau, 5000)

	dm_X = distance.cdist(pc, pc)

	# Maxmin subsampling

	ind_L, cover_r = cqpd.maxmin(dm_X, 200)

	dm_L = dm_X[ind_L,:][:,ind_L]

	# Persistent homology

	q = 2

	result = ripser(dm_L, coeff=q, distance_matrix=True, maxdim=2, do_cocycles=True)

	# result = ripser(pc, coeff=q, maxdim=2, do_cocycles=True, n_perm=300)


	diagrams = result['dgms']
	cocycles = result['cocycles']

	plt.close()
	plot_diagrams(diagrams)
	plt.savefig('roc_curves/quasi_persistence_{}.png'.format(i))
	plt.close()

	co1_ind=-1
	co2_ind=-2

	cup_bar = cqpd.getQuasiPeriodicScore(diagrams, cocycles, dm_L, coeff=q, cocycle_1_ind=co1_ind, cocycle_2_ind=co2_ind)

	cup.append(cup_bar)
	real_quasi.append(1)

	# Plot new diagrams

	# Plot of persistent diagram + cup product
	plt.close()
	plt.scatter(diagrams[0][:,0], diagrams[0][:,1], label='H_0')
	plt.scatter(diagrams[1][:,0], diagrams[1][:,1], label='H_1')
	plt.scatter(diagrams[2][:,0], diagrams[2][:,1], label='H_2')

	plt.scatter(cup_bar[0], cup_bar[1], label ='Cup')

	H_1 = cocycles[1]
	H_1_diagram = diagrams[1]
	H_1_persistence = H_1_diagram[:,1] - H_1_diagram[:,0]
	H_1_persistence_sort_ind = H_1_persistence.argsort()

	second_h1.append(H_1_diagram[H_1_persistence_sort_ind[-2],:])

	chosen = diagrams[1][[H_1_persistence_sort_ind[co1_ind], H_1_persistence_sort_ind[co2_ind]],:]

	plt.scatter(chosen[:,0], chosen[:,1], label='Selected classes', facecolors='none', edgecolors='r')

	plt.legend(loc = 'lower right')

	plt.savefig('roc_curves/quasi_cup_persistence_{}.png'.format(i))
	plt.close()

	H_2 = cocycles[2]
	H_2_diagram = diagrams[2]
	H_2_persistence = H_2_diagram[:,1] - H_2_diagram[:,0]
	H_2_persistence_sort_ind = H_2_persistence.argsort()

	if H_2_diagram.shape[0] < 1:
		first_h2.append(np.array([0,0]))
	else:
		first_h2.append(H_2_diagram[H_2_persistence_sort_ind[-1],:])

	print(i)


# -----------------------------------------------------------------------------
# Periodic
# -----------------------------------------------------------------------------
print('Periodic')
for i in range(10):

	rng = np.random.default_rng()

	A = 1
	B = rng.integers(low=1, high=5)
	f1 = rng.integers(low=1, high=5)
	f2 = rng.integers(low=1, high=5)

	time_series = f(t, A, B, f1, f2)

	plt.plot(t, time_series)
	plt.title(r'$f(t) = {}sin({}t) + {}cos({}t)$'.format(A,B,f1,f2))
	plt.savefig('roc_curves/perio_plot_{}.png'.format(i))
	plt.close()

	tau, d = cqpd.optimal_tau_frequencies(time_series, t, np.sort([f1,-f1,f2,-f2]))

	pc = cqpd.SW(f, t, d, tau)

	# pc = cqpd.sw_map(time_series, t, d, tau, 5000)

	dm_X = distance.cdist(pc, pc)

	# Maxmin subsampling

	ind_L, cover_r = cqpd.maxmin(dm_X, 200)

	dm_L = dm_X[ind_L,:][:,ind_L]

	# Persistent homology

	q = 2

	result = ripser(dm_L, coeff=q, distance_matrix=True, maxdim=2, do_cocycles=True)

	# result = ripser(pc, coeff=q, maxdim=2, do_cocycles=True, n_perm=300)


	diagrams = result['dgms']
	cocycles = result['cocycles']

	plt.close()
	plot_diagrams(diagrams)
	plt.savefig('roc_curves/perio_persistence_{}.png'.format(i))
	plt.close()

	co1_ind=-1
	co2_ind=-2

	cup_bar = cqpd.getQuasiPeriodicScore(diagrams, cocycles, dm_L, coeff=q, cocycle_1_ind=co1_ind, cocycle_2_ind=co2_ind)

	cup.append(cup_bar)
	real_quasi.append(0)

	# Plot new diagrams

	# Plot of persistent diagram + cup product
	plt.close()
	plt.scatter(diagrams[0][:,0], diagrams[0][:,1], label='H_0')
	plt.scatter(diagrams[1][:,0], diagrams[1][:,1], label='H_1')
	plt.scatter(diagrams[2][:,0], diagrams[2][:,1], label='H_2')

	plt.scatter(cup_bar[0], cup_bar[1], label ='Cup')

	H_1 = cocycles[1]
	H_1_diagram = diagrams[1]
	H_1_persistence = H_1_diagram[:,1] - H_1_diagram[:,0]
	H_1_persistence_sort_ind = H_1_persistence.argsort()

	second_h1.append(H_1_diagram[H_1_persistence_sort_ind[-2],:])

	chosen = diagrams[1][[H_1_persistence_sort_ind[co1_ind], H_1_persistence_sort_ind[co2_ind]],:]

	plt.scatter(chosen[:,0], chosen[:,1], label='Selected classes', facecolors='none', edgecolors='r')

	plt.legend(loc = 'lower right')

	plt.savefig('roc_curves/perio_cup_persistence_{}.png'.format(i))
	plt.close()

	H_2 = cocycles[2]
	H_2_diagram = diagrams[2]
	H_2_persistence = H_2_diagram[:,1] - H_2_diagram[:,0]
	H_2_persistence_sort_ind = H_2_persistence.argsort()

	if H_2_diagram.shape[0] < 1:
		first_h2.append(np.array([0,0]))
	else:
		first_h2.append(H_2_diagram[H_2_persistence_sort_ind[-1],:])

	print(i)





final_results = np.column_stack((first_h2, second_h1, cup, real_quasi))

np.savetxt("roc_curves/results.csv", final_results, delimiter=",")

# Compute ROC

cup_scores = np.sqrt(np.absolute(final_results[:,2] - final_results[:,3])*np.absolute(final_results[:,4] - final_results[:,5])/3)

original_scores = np.sqrt(np.absolute(final_results[:,0] - final_results[:,1])*np.absolute(final_results[:,2] - final_results[:,3])/3)

np.savetxt("roc_curves/scores.csv", np.column_stack((original_scores, cup_scores)), delimiter=",")

# Original labels
y = final_results[:,6]

# Binarized labels
binarize_y = label_binarize(y, classes=[0,1])
print(binarize_y.shape)

# Compute AUC for each label
auc_cup = []
auc_original = []
for j in range(binarize_y.shape[1]):

	y = binarize_y[:,j]

	cup_fpr, cup_tpr, tresh = metrics.roc_curve(y,cup_scores)
	auc_cup.append( metrics.auc(cup_fpr, cup_tpr) )

	original_fpr, original_tpr, tresh = metrics.roc_curve(y,original_scores)
	auc_original.append( metrics.auc(original_fpr, original_tpr) )

	plt.close()
	plt.figure(1)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(cup_fpr, cup_tpr, label='Cup')
	plt.plot(original_fpr, original_tpr, label='Original')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend(loc=4)
	plt.savefig('roc_curves/roc_{}.png'.format(j))
	plt.close()

AUC_compared = np.row_stack((auc_cup, auc_original))

np.savetxt("roc_curves/auc_results.csv", AUC_compared, delimiter=",")
