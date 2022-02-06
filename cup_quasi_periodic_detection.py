import numpy as np
import math
import itertools
import sys
import matplotlib.pyplot as plt
import copy
import scipy.signal
import numpy.matlib
import gc

from numpy import linalg as LA
from scipy.interpolate import UnivariateSpline
from mpl_toolkits import mplot3d
from ripser import ripser
from persim import plot_diagrams
from scipy.spatial import distance_matrix
from scipy.spatial import distance

def gen_sphere(N,R):
    theta = np.random.uniform(0, 2*math.pi, N)
    phi = np.random.uniform(0, 2*math.pi, N)
    
    x = np.multiply( (R*np.cos(theta)), np.cos(phi))
    y = np.multiply( (R*np.cos(theta)), np.sin(phi))
    z = R*np.sin(theta)

    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    torus = np.column_stack((x,y,z))
    return torus

def gen_torus(N, R, r):
    theta = np.random.uniform(0, 2*math.pi, N)
    phi = np.random.uniform(0, 2*math.pi, N)
    
    x = np.multiply( (R + r*np.cos(theta)), np.cos(phi))
    y = np.multiply( (R + r*np.cos(theta)), np.sin(phi))
    z = r*np.sin(theta)

    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    torus = np.column_stack((x,y,z))
    return torus

def gen_flat_torus(N):
    X = np.random.uniform(0,1, (N,2))

    dm_X = distance.cdist(X,X)

    for i in range(-1,2):
        for j in range(-1,2):
            X1 = copy.deepcopy(X)
            X1[:,0] = X1[:,0] + i
            X1[:,1] = X1[:,1] + j
            
            dm1 = distance.cdist(X,X1)
            
            dm_X = np.minimum(dm_X, dm1)

    return dm_X


def minmax_subsample_point_cloud(X, num_landmarks, distance):
    """
    This function computes minmax subsampling using point cloud and a distance function.

    :type X: numpy array
    :param X: Point cloud. If X is a nxm matrix, then we are working with a pointcloud with n points and m variables.

    :type num_landmarks: int
    :param num_landmarks: Number of landmarks

    :type distance: function
    :param  distance: Distance function. Must be able to compute distance between 2 point cloud with same dimension and different number of points in each point cloud.
    """
    num_points = len(X)
    ind_L = [np.random.randint(0,num_points)]  

    distance_to_L = distance(X[ind_L,:], X)

    for i in range(num_landmarks-1):
        ind_max = np.argmax(distance_to_L)
        ind_L.append(ind_max)
        
        dist_temp = distance(X[[ind_max],:], X)

        distance_to_L = np.minimum(distance_to_L, dist_temp)

    return {'indices':ind_L, 'distance_to_L':distance_to_L}

def maxmin(dist_matrix, n):
    '''
    Given a distance matrix retunrs a maxmin subsampling and the covering radious 
    corresponding to the subsampled set.
    
    :param dist_matrix: Distance matrix
    :param n: Size of subsample set.
    :returns L: List of indices corresponding to the subsample set.
    :return cr: Covering radious for the subsample set.
    '''
    L = [np.random.randint(0,len(dist_matrix))]
    
    dist_to_L = dist_matrix[ L[-1] ,:]
    
    for i in range(n-1):
        ind = np.argmax( dist_to_L )
        L.append(ind)
        
        dist_to_L = np.minimum(dist_to_L, dist_matrix[L[-1], :])
        
    cr = np.max(dist_to_L)

    return L, cr

def sw_map(f, t, d, tau, num_points):
    '''
    This function computes the sliding window embeding of a given signal f.
    
    :param f: signal represented as a 1D array.
    :param t: time variable represented as a 1D array.
    :param w: window size
    :param d: d + 1 will be the dimension of the embedding
    :param num_points: number of points in the final sliding window point cloud 
    '''
    
    w = tau*d
        
    # Step 1 :
    
    T =  np.linspace(t[0], t[-1]-w,num_points)
    
    tt = np.dot(np.ones((d+1,1)), T.reshape(1,-1)) + tau*np.dot(np.arange(d+1).reshape(-1,1), np.ones((1,num_points))) 
    
    
    # Step 2 :
    
    interpolation = UnivariateSpline(t, f, s=0)
    
    point_cloud = np.apply_along_axis(interpolation, 1, tt)
    
    return point_cloud.transpose()

def SW(my_f, T, d, tau):
    time_T = np.matlib.repmat(T,d+1,1).T
    delay = tau*np.arange(0,d+1)
    SW_f =my_f( time_T +  delay)
    if LA.norm(np.imag(SW_f)) >0 :
        SW_f = np.concatenate((np.real(SW_f), np.imag(SW_f)), axis = 1)
    return SW_f

def optimal_tau2(my_f, t):
    
    time_series = my_f
    
    n_t = len(t)

    delta_t = (t[-1] - t[0])/n_t

    f_hat = np.fft.fft(time_series)
        
    abs_f_hat = np.abs(f_hat)
    abs_f_hat = abs_f_hat/LA.norm(abs_f_hat[0:n_t//2])

    ind, _ = scipy.signal.find_peaks(abs_f_hat[1:n_t//2], height=.1)

    freq = np.fft.fftfreq(n_t, d=delta_t/(2*np.pi))
    
    ind += 1 
    
    k_dot_omega = freq[ind]

    if LA.norm(np.imag(time_series)) == 0:
        k_dot_omega = np.concatenate((k_dot_omega, -k_dot_omega), axis=None)

    x = t[0:len(t)//2]
    
    if x[0] == 0:
        x = x[1:]
    
    d = len(k_dot_omega)
    k_dot_omega_diff = np.zeros(d*(d-1)//2)
    
    k = 0
    
    for i in range(d):
        for j in range(i+1,d):
            k_dot_omega_diff[k] = k_dot_omega[i] - k_dot_omega[j]
            k += 1
    
    diff_dot_omega = k_dot_omega_diff.reshape((-1,1))
    
    arg = np.dot(diff_dot_omega , x.reshape((1,-1)))
    
    G = (1 - np.exp(1j*arg*(d+1)))/(1 - np.exp(1j*arg))
    
    G = np.sum(np.abs(G)**2, axis=0)
    
    plt.plot(x,G)
    
    tau = x[np.argmin(G)]

    return tau, d

def optimal_tau_frequencies(my_f, t, frequencies):
    
    k_dot_omega = frequencies

    x = t[0:len(t)//2]
    
    if x[0] == 0:
        x = x[1:]
    
    d = len(k_dot_omega)
    k_dot_omega_diff = np.zeros(d*(d-1)//2)
    
    k = 0
    
    for i in range(d):
        for j in range(i+1,d):
            k_dot_omega_diff[k] = k_dot_omega[i] - k_dot_omega[j]
            k += 1
    
    diff_dot_omega = k_dot_omega_diff.reshape((-1,1))
    
    arg = np.dot(diff_dot_omega , x.reshape((1,-1)))
    
    G = (1 - np.exp(1j*arg*(d+1)))/(1 - np.exp(1j*arg))
    
    G = np.sum(np.abs(G)**2, axis=0)
    
    plt.plot(x,G)
    
    tau = x[np.argmin(G)]

    return tau, d    

def cup_product_cochains(cocycle_1, cocycle_2, q):
    cup = []
    for simplex1 in cocycle_1:
        end = simplex1[1]
        for simplex2 in cocycle_2:
            start = simplex2[0]
            if end == start:
                cup.append([simplex1[0], simplex1[1], simplex2[1], np.remainder(simplex1[2]*simplex2[2], q)])

    return cup

def rips_complex(dist_m, alpha):
    '''
    This function computes the 0, 1 and 2-skeleton of the Rips filtration. 

    Parameters
    ----------
    dist_m: np.array
            Distance matrix.

    alpha:  float
            maximum scale to compute the Rips complex filtration.

    Returns
    -------
    R:  dictionary
        Dictionary containig the 0, 1 and 2-skeletonm of the Rips filtration. The simplices in each skeleton are ordered by decreasing diameter.
    '''
    
    D = copy.deepcopy(dist_m)
    
    # Rips complex
    R = {}
    # Orders
    order = {}

    diameters = {}

    # 0-simplices
    R['0'] = list(np.arange(0,len(dist_m)))
    order['0'] = np.arange(0,len(dist_m))
    diameters['0'] = []
        
    # D[D > alpha] = 0

    non_zero = np.nonzero(D)


    # 1-simplices
    R['1'] = []
    diameter_1 = []
    order_1 = []
    for j in itertools.combinations( np.arange(len(dist_m)), 2 ):
        ind = list(np.array(j))
        
        R['1'].append(ind)
        # diameter_1.append(np.max(dist_m[ind,:][:,ind]))
        diameter_1.append(np.max(D[ind,:][:,ind]))

    R['1'] = np.array(R['1'])

    # order_1 = np.argsort(diameter_1, kind='mergesort')[::-1]
    order_1 = np.argsort(diameter_1, kind='mergesort')

    R['1'] = R['1'][order_1]
    
    order['1'] = order_1[order_1]

    diameters['1'] = np.array(diameter_1)[order_1]

    # droping all the simplices with diameter larger than alpha

    R['1'] = R['1'][diameters['1'] <= alpha]

    order['1'] = order['1'][diameters['1'] <= alpha]

    diameters['1'] = diameters['1'][diameters['1'] <= alpha]

    

    # 2-simplices
    R['2'] = []
    diameter_2 = []
    order_2 = []
    for j in itertools.combinations( np.arange(len(dist_m)), 3 ):
        ind = list(np.array(j))
        
        R['2'].append(ind)
        # diameter_2.append(np.max(dist_m[ind,:][:,ind]))
        diameter_2.append(np.max(D[ind,:][:,ind]))

    R['2'] = np.array(R['2'])

    # order_2 = np.argsort(diameter_2, kind='mergesort')[::-1]
    order_2 = np.argsort(diameter_2, kind='mergesort')

    R['2'] = R['2'][order_2]

    order['2'] = order_2[order_2]

    diameters['2'] = np.array(diameter_2)[order_2]

    # droping all the simplices with diameter larger than alpha

    R['2'] = R['2'][diameters['2'] <= alpha]

    order['2'] = order_2[diameters['2'] <= alpha]

    diameters['2'] = diameters['2'][diameters['2'] <= alpha]

    
    return R, order, diameters


def delta_0(simplex_0, simplex_1):
    '''
    This function computes coboundary matric delta_0.

    Parameters
    ----------
    simplex_0:  np.array
                0-skeleton of the Rips filtration

    simplex_1:  np.array
                1-skeleton of the Rips filtration

    Returns
    -------
    Coboundary matrix delta_0.
    '''

    num_0_sim = len(simplex_0)
    num_1_sim = len(simplex_1)
    
    partial_0 = np.zeros((num_0_sim, num_1_sim))
    
    for i in range(num_1_sim):
        sigma = simplex_1[i]
        partial_0[sigma, i] = 1

    return np.transpose(partial_0[::-1,::-1]) # return anti-transpose
    # return np.transpose(partial_0)


def delta_1(simplex_1, simplex_2):
    '''
    This function computes coboundary matric delta_1.

    Parameters
    ----------
    simplex_1:  np.array
                1-skeleton of the Rips filtration

    simplex_2:  np.array
                2-skeleton of the Rips filtration

    Returns
    -------
    Coboundary matrix delta_1.
    '''

    num_1_sim = len(simplex_1)
    num_2_sim = len(simplex_2)
    
    partial_1 = np.zeros((num_1_sim, num_2_sim))
    
    for i in range(num_2_sim):

        sigma = simplex_2[i]

        for j in range(3):
            sigma_copy = copy.deepcopy(sigma)
            
            sigma_copy = np.delete(sigma_copy, j)
            
            face = sigma_copy

            row = np.where((simplex_1 == face).all(axis=1))

            partial_1[row,i] = 1
        
    return np.transpose(partial_1[::-1,::-1]) # return anti-transpose
    # return np.transpose(partial_1)

def matrix_reduction(D):

    # number of rows in D
    n_rows = D.shape[0]
    # number of columns in D
    n_cols = D.shape[1]

    # Initialize reduced matrix R
    R = copy.deepcopy(D)
    # Initialize elementary operations matrix
    V = np.eye(n_cols)

    # Initialize lowest ones posiitons as -1
    low = -np.ones(n_cols, dtype=int)

    # Find the lowest one entry en each column of D
    for k in range(n_cols):
        col = R[:,k]
        if np.nonzero(col)[-1].size > 0:
            low[k] = np.nonzero(col)[-1][-1]
        else: 
            low[k] = -1
    # print(low)

    for current_low in range(n_rows-1,-1,-1):
        current_columns = np.where(low == current_low)[-1]
        # print('cur-columns = ',current_columns)

        if current_columns.size > 1:
            i = current_columns[0]
            # print('i=',i)
            columns_to_reduce = current_columns[1:]

            for j in columns_to_reduce:
                # print('j=',j)
                R[:,j] = np.remainder(R[:,j] - R[:,i], 2)
                V[:,j] = np.remainder(V[:,j] - V[:,i], 2)
                
                if np.nonzero(R[:,j])[-1].size > 0:
                    low[j] = np.nonzero(R[:,j])[-1][-1]
                else:
                    low[j] = -1

                # print(low)
                
                # print(R)

                # print('')

    return R, V, low

def backwards_substitution(A,low,b):
    
    if A.shape[0] != b.shape[0]:
        raise ValueError('Incompatible dimensions between matrix A and vector b in the equation Ax = b.')
    # number of rows
    n_rows = A.shape[0]
    # initialize solution vector
    x = np.zeros(A.shape[1])

    for i in range(n_rows-1,-1,-1):
        # check if i-row in A is zero
        if sum(A[i,:]) == 0:
            # if corresponding entry in b is non-zero: inconsistent system. Break loop and return values
            if b[i] != 0:
                return x, i
            # if corresponding entry in b is zero: we do not need to modify solution x. Just continue iterations
            else:
                continue
        # if i-row in A is non-zero
        else:
            # If current row is a lowest one: backwards substitution
            if np.where(low == i)[0].size > 0:
                j = np.where(low == i)[0][0]

                # This is a more memory intensive solution that is correct. USE TO CHECK CORRECTNESS!

                # temp_row = copy.deepcopy(A[i,:])
                # temp_x = copy.deepcopy(x)
                
                # temp_row = np.delete(temp_row, j)
                # temp_x = np.delete(temp_x, j)
                
                # x[j] = np.remainder(b[i] + np.remainder(np.dot(temp_row, temp_x), 2), 2)

                # ALTERNATIVE SOLUTION: Here we are using the fact that in the current step x[j] = 0 since we have not modified this entry in the BS-algorithm.

                x[j] = np.remainder(b[i] + np.dot(A[i,:], x), 2)

            # If current row is not a lowest one: check for consistency
            else:
                dot_product = np.remainder(np.dot(A[i,:], x), 2)
                # If system is consistent continue
                if dot_product == b[i]:
                    continue
                # If system is inconsistent, break loop and return values
                else:
                    return x, i

    # if i == 0:
    return x, -1

def getQuasiPeriodicScore(diagrams, cocycles, distance_matrix, coeff=2, cocycle_1_ind=-1, cocycle_2_ind=-2):
    # H_1 processing and sorting

    H_1 = cocycles[1]
    H_1_diagram = diagrams[1]
    H_1_persistence = H_1_diagram[:,1] - H_1_diagram[:,0]
    H_1_persistence_sort_ind = H_1_persistence.argsort()

    # Cup product 

    #Lets handle if H_1 is empty

    if H_1_diagram.shape[0] < 1:
        return np.array([0, 0])

    cocycle_1 = H_1[H_1_persistence_sort_ind[cocycle_1_ind]]
    cocycle_2 = H_1[H_1_persistence_sort_ind[cocycle_2_ind]]

    cup = cup_product_cochains(cocycle_1, cocycle_2, coeff)

    # Compute treshhold using H_1

    temp = np.sort(H_1_diagram[:,1])
    treshhold = temp[-2]

    # Rips complex

    rips_com, orders, diameters = rips_complex(distance_matrix, treshhold)

    # Coboundary matrices

    d0 = delta_0(rips_com['0'], rips_com['1'])
    d1 = delta_1(rips_com['1'], rips_com['2'])

    # delta^1 reducition

    R1, V1, low = matrix_reduction(d1)

    # Cup product as a cochain 

    cochain = np.zeros(rips_com['2'].shape[0])

    cup = np.array(cup)

    for i in range(len(cup)):
        simplex = cup[i,2::-1]
        
        j = np.where((rips_com['2'] == simplex).all(axis=1))
        
        cochain[j] = cup[i,3]
        
    cochain = cochain[::-1]

    # Backwards substitution

    y, index = backwards_substitution(R1, low, cochain)

    # Cohomological death
    if index != -1:
        ind_goal = rips_com['2'][len(rips_com['2'])-index-1]
    else:
        ind_goal = rips_com['2'][-1]

    death = np.max(distance_matrix[np.ix_(ind_goal[::-1],ind_goal[::-1])])

    # Cohomological birth: as minimum of birth between classes

    birth = min(H_1_diagram[H_1_persistence_sort_ind[cocycle_1_ind]][1], H_1_diagram[H_1_persistence_sort_ind[cocycle_2_ind]][1])

    # score = np.sqrt( ( H_1_persistence[H_1_persistence_sort_ind[-2]] * (homology_death - homology_birth) )/3)

    # Clear memory
    del rips_com
    del orders
    del diameters
    del d0
    del d1
    del R1
    del V1
    del low
    del cochain

    return np.array([death, birth])