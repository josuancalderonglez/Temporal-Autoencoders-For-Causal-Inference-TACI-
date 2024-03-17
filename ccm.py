import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # <--- This is important for 3d plotting
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from statsmodels.tsa import stattools
from sklearn.metrics import mutual_info_score
from scipy.spatial import distance
from scipy.spatial import cKDTree as KDTree

import skccm

class ccm_causality:
        
    def CCM_pairwise(variable_x, variable_y, lag, embed_dim, lib_lens_test=True):

        e1 = skccm.Embed(variable_x)
        X1 = e1.embed_vectors_1d(lag,embed_dim)

        e2 = skccm.Embed(variable_y)
        X2 = e2.embed_vectors_1d(lag,embed_dim)

        from skccm.utilities import train_test_split

        #split the embedded time series
        x1tr, x1te, x2tr, x2te = train_test_split(X1,X2, percent=.75)

        ccm = skccm.CCM() #initiate the class

        if lib_lens_test:
            #library lengths to test
            len_tr = len(x1tr)
            lib_lens = np.arange(100, len_tr, len_tr/100, dtype='int')
        else:
            lib_lens = [len(x1tr)]

        #test causation
        ccm.fit(x1tr,x2tr)
        x1p, x2p = ccm.predict(x1te, x2te,lib_lengths=lib_lens)

        sc1, sc2 = ccm.score()

        return sc2, sc1  
    
    def saturation_pt_curve(curve,saturation_pt = 7, threshold = 1e-2):

        ind_der_0 = np.where(np.abs(np.gradient(curve))<threshold)

        if len(ind_der_0) >= 2: 
            saturation_pt = 7

            consec = 0

            for x,y in zip(ind_der_0[:],ind_der_0[1:]):

                if x == y - 1:
                    consec += 1
                else:
                    consec = 0

                if consec == saturation_pt:
                    convergence_pt = x-int(saturation_pt/2)
                    break

        if (len(ind_der_0) < 2) or (x == ind_der_0[-2]):
            convergence_pt = np.argmin(abs(curve - np.mean(curve[-20])))
                
        return convergence_pt
    
    def localmin(x):
        """Return all local minima from the given data set.
        Returns all local minima from the given data set.  Note that even
        "kinky" minima (which are probably not real minima) will be
        returned.
        Parameters
        ----------
        x : array
            1D scalar data set.
        Returns
        -------
        i : array
            Array containing location of all local minima.
        """
        return (np.diff(np.sign(np.diff(x))) > 0).nonzero()[0] + 1

    def calc_MI(x, lags, bins):
            
        mi=[]
        for lag in lags:
            c_xy = np.histogram2d(x[:-lag], x[lag:], bins)[0]
            mi.append(mutual_info_score(None, None, contingency=c_xy))
        return mi
    
    def mut_info_plot(variables,var_names,maxtau,plot_graph,print_out):

        lags=list(range(1,maxtau+1))
        loc_minimum=[]
        lag_val=[]
        mutual_info=np.zeros((variables.shape[0],len(lags)))

        for ii in range(variables.shape[0]):

            mutual_info[ii,:]=ccm_causality.calc_MI(variables[ii],lags,bins=64)
            loc_minimum.append(ccm_causality.localmin(mutual_info[ii,:])[0])            
            lag_val.append(lags[loc_minimum[ii]])
            
        if plot_graph:
            plt.figure()
            for ii in range(variables.shape[0]):
                plt.subplot(1, variables.shape[0], ii+1)
                plt.plot(lags,mutual_info[ii,:])
                plt.plot(lags[loc_minimum[ii]],
                         mutual_info[ii,loc_minimum[ii]],'r*')
            #     plt.xlabel('Lags')
            #     plt.ylabel('Mutual Information')
                plt.xticks(np.arange(0, len(lags), 50))
                plt.yticks([])
                plt.grid()

            plt.suptitle("Lag Value Mutual Information")
            plt.show()
        
        if print_out:
            print('The lag values according to the mutual information are:')

            for ii in range(variables.shape[0]):
                print(str(var_names[ii]),'->',
                      lags[loc_minimum[ii]])

        return lag_val
    
    def neighbors(y, metric='chebyshev', window=0, maxnum=None):
        """Find nearest neighbors of all points in the given array.
        Finds the nearest neighbors of all points in the given array using
        SciPy's KDTree search.
        Parameters
        ----------
        y : ndarray
            N-dimensional array containing time-delayed vectors.
        metric : string, optional (default = 'chebyshev')
            Metric to use for distance computation.  Must be one of
            "cityblock" (aka the Manhattan metric), "chebyshev" (aka the
            maximum norm metric), or "euclidean".
        window : int, optional (default = 0)
            Minimum temporal separation (Theiler window) that should exist
            between near neighbors.  This is crucial while computing
            Lyapunov exponents and the correlation dimension.
        maxnum : int, optional (default = None (optimum))
            Maximum number of near neighbors that should be found for each
            point.  In rare cases, when there are no neighbors that are at a
            nonzero distance, this will have to be increased (i.e., beyond
            2 * window + 3).
        Returns
        -------
        index : array
            Array containing indices of near neighbors.
        dist : array
            Array containing near neighbor distances.
        """
        if metric == 'cityblock':
            p = 1
        elif metric == 'euclidean':
            p = 2
        elif metric == 'chebyshev':
            p = np.inf
        else:
            raise ValueError('Unknown metric.  Should be one of "cityblock", '
                             '"euclidean", or "chebyshev".')

        tree = KDTree(y)
        n = len(y)

        if not maxnum:
            maxnum = (window + 1) + 1 + (window + 1)
        else:
            maxnum = max(1, maxnum)

        if maxnum >= n:
            raise ValueError('maxnum is bigger than array length.')

        dists = np.empty(n)
        indices = np.empty(n, dtype=int)

        for i, x in enumerate(y):
            for k in range(2, maxnum + 2):
                dist, index = tree.query(x, k=k, p=p)
                valid = (np.abs(index - i) > window) & (dist > 0)

                if np.count_nonzero(valid):
                    dists[i] = dist[valid][0]
                    indices[i] = index[valid][0]
                    break

                if k == (maxnum + 1):
                    raise Exception('Could not find any near neighbor with a '
                                    'nonzero distance.  Try increasing the '
                                    'value of maxnum.')

        return np.squeeze(indices), np.squeeze(dists)

    def fnn(x, dim, tau, Rtol, Ra, metric,maxnum, window):
        """Return fraction of false nearest neighbors for a single d.
        Returns the fraction of false nearest neighbors for a single d.
        This function is meant to be called from the main fnn() function.
        See the docstring of fnn() for more.
        """
        # We need to reduce the number of points in dimension d by tau
        # so that after reconstruction, there'll be equal number of points
        # at both dimension d as well as dimension d + 1.
        e1 = skccm.Embed(x[:-tau])
        y1 = e1.embed_vectors_1d(tau,dim)

        e2 = skccm.Embed(x)
        y2 = e2.embed_vectors_1d(tau,dim + 1)

        # Find near neighbors in dimension d.
        if not maxnum:
            maxnum = y1.shape[0]
        else:
            maxnum = max(1, maxnum)

        neigh = NearestNeighbors(n_neighbors=maxnum, metric='euclidean')
        neigh.fit(y1) 
        dist,index=neigh.kneighbors(y1)
        dist=dist[:,1]
        index=index[:,1]
        
        if np.sum(np.array(dist)==0)!=0:
            raise Exception('Could not find any near neighbor with a '
                                'nonzero distance.')
        
#         # Find near neighbors in dimension d.
#         index, dist = causality.neighbors(y1, metric=metric, window=window,
#                                   maxnum=maxnum)
        
        # Find all potential false neighbors using Kennel et al.'s tests.
        f1 = np.abs(y2[:, -1] - y2[index, -1]) / dist > Rtol
        f2=np.asarray([distance.euclidean(i, j) for i, j in zip(y2, y2[index])])/ np.std(x) > Ra

        f3 = f1 | f2

        return np.mean(f1), np.mean(f2), np.mean(f3)

    def false_nearest_neigh(x, tau, max_dim, max_num_neigh, plot_fig=False):

        try:
            stopping_threshold=3

            fnn1=[]
            fnn2=[]

            for dd in range(1,max_dim):

                f1,f2,f3=ccm_causality.fnn(x, dim = dd, tau = tau, Rtol = 15.0, Ra = 2.0, 
                                       metric='euclidean',maxnum = max_num_neigh, window =10)

                fnn1.append(f1*100)
                fnn2.append(f2*100)

                if f1*100<stopping_threshold:
                    break


            embed_dim = dd
        except:
            embed_dim = 1
        
        if plot_fig:
            plt.figure()
            plt.plot(list(range(1,dd+1)),fnn1,'b')
            plt.plot(list(range(1,dd+1)),fnn2,'r')
            plt.ylim([0, 100])
            plt.xticks(np.arange(1, dd+1,1))
            plt.show()

        return embed_dim
    
    def emb_dim_fnn(variables, lags, max_dim, max_num_neigh, plot_graph, print_out):
    
        tau=[]
        embed_dim=[]

        for ii in range(variables.shape[0]):

            tau.append(lags[ii])

            embed_dim.append(ccm_causality.false_nearest_neigh(variables[ii], tau=tau[ii], 
                                                           max_dim = max_dim, 
                                                           max_num_neigh = max_num_neigh, 
                                                           plot_fig=plot_graph))

        if print_out:

            print('The embedding dimensions are:')

            for ii in range(variables.shape[0]):

                print('variable',str(var_names[ii]),' = ',str(embed_dim[ii]))

        return tau, embed_dim