# -*- coding: utf-8 -*-

# http://codereview.stackexchange.com/questions/61598/k-mean-with-numpy

import numpy as np
import scipy as sp
import heapq
from collections import Counter

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity, linear_kernel
from sklearn.cluster import KMeans, MiniBatchKMeans

def reduce_norm(vec, reduction=0.95):
    if (vec != 0).sum() <= 5:
        return vec

    norm2 = vec.dot(vec)

    # using squared norm, so use the square of reduction coeff
    desired_norm2 = norm2 * (reduction ** 2)

    for idx in vec.argsort():
        if norm2 <= desired_norm2:
            break

        norm2 = norm2 - vec[idx] ** 2
        vec[idx] = 0.0

    return vec


def cluster_centroids(X, clusters, k):
    result = []
    cluster_no = 0

    for i in xrange(k):
        if (clusters == i).sum() == 0:
            continue

        centroid = X[clusters == i].mean(axis=0)
        if sp.sparse.issparse(centroid):
            centroid = centroid.toarray().flatten()

        # centroid = reduce_norm(centroid)
        result.append(centroid)
        cluster_no = cluster_no + 1

    return np.array(result), cluster_no



def kmeans_objective(X, C, labels):
    J = 0.0
    k = labels.max()

    for i in xrange(k + 1):
        if (labels == i).sum() == 0:
            continue
        X_i = X[labels == i]
        centroid = C[i, :]
        D = euclidean_distances(X_i, centroid, squared=0)
        J = J + D.sum()

    return J

def kmeans(X, k, centroids=None, steps=20, verbose=0):
    if not centroids:
        centroids = X[np.random.choice(np.arange(X.shape[0]), size=k)]

        if sp.sparse.issparse(centroids):
            centroids = centroids.toarray()

    for step in xrange(steps):
        D = euclidean_distances(centroids, X, squared=0) # since rows are normalized, it's cosine
        clusters = D.argmin(axis=0)

        new_centroids, k = cluster_centroids(X, clusters, k)
        
        J = np.abs((new_centroids ** 2).sum() - (centroids ** 2).sum())
        if verbose and step % 10 == 0:
            print 'step %d... J=%0.4f' % (step, J)

        if J < 1e-6:
            break

        centroids = new_centroids

    if verbose:
        print 'converged after step=%d, final J=%0.4f' % (step, J)
    D = euclidean_distances(centroids, X, squared=0)
    clusters = D.argmin(axis=0)
    return clusters, k, centroids


def bisecting_kmeans(X, k, steps=20, verbose=0):
    N, _ = X.shape
    labels = np.zeros(N, dtype=np.int)

    sizes_heap = []
    heapq.heappush(sizes_heap, (-N, 0))
    k_max = 1

    while k_max < k:
        if verbose and k_max % 50 == 0:
            print 'iteration %d...' % k_max

        size, idx = heapq.heappop(sizes_heap)
        size = -size

        J_min = 1e60
        best_labels = None
        it_labels = None
        trials = 10
  
        while trials > 0:
            it_labels, k_new, C = kmeans(X[labels == idx], k=2, steps=steps, verbose=0)

            J_it = kmeans_objective(X[labels == idx], C, it_labels)
            if (it_labels == 1).sum() < 2 or (it_labels == 0).sum() < 2:
                continue
            
            if J_it < J_min and k_new == 2:
                J_min = J_it
                best_labels = it_labels

            trials = trials - 1

        if best_labels is None:
            if it_labels is None:
                # only if some cluster has 1 or fewer elements 
                if verbose:
                    print 'discarding cluster of size %d at index %d as it cannot be bisected' % (size, idx)
                continue
            else:
                best_labels = it_labels
                J_min = J_it

        ones_size = (best_labels == 1).sum()
        heapq.heappush(sizes_heap, (-ones_size, k_max))
        heapq.heappush(sizes_heap, (-(size - ones_size), idx))

        best_labels[best_labels == 1] = k_max
        best_labels[best_labels == 0] = idx
        
        labels[labels == idx] = best_labels
    
        if verbose:
            print 'iteration %d... bisected cluster %d (of size %d) into %d and %d, best J_min=%0.5f' % \
                    (k_max, idx, size, ones_size, size - ones_size, J_min)
        k_max = k_max + 1
        

    return labels


def sklearn_bisecting_kmeans(X, k, verbose=0):
    labels, _ = sklearn_bisecting_kmeans_lineage(X, k, verbose)
    return labels

def sklearn_bisecting_kmeans_lineage(X, k, verbose=0):
    N, _ = X.shape
    labels = np.zeros(N, dtype=np.int)
    lineage = np.zeros((k, N), dtype=np.int)

    sizes_heap = []
    heapq.heappush(sizes_heap, (-N, 0))
    k_max = 1

    
    while k_max < k:
        if verbose and k_max % 50 == 0:
            print 'iteration %d...' % k_max

        size, idx = heapq.heappop(sizes_heap)
        size = -size

        trials = 3
        while trials > 0:
            if size < 100:
                model = KMeans(n_clusters=2)
            else:
                model = MiniBatchKMeans(n_clusters=2, init='random')

            km = model.fit(X[labels == idx])

            it_labels = km.labels_
            if (it_labels == 1).sum() > 1 and (it_labels == 0).sum() > 1:
                break
            trials = trials - 1

        ones_size = (it_labels == 1).sum()
        heapq.heappush(sizes_heap, (-ones_size, k_max))
        heapq.heappush(sizes_heap, (-(size - ones_size), idx))
        
        it_labels[it_labels == 1] = k_max
        it_labels[it_labels == 0] = idx        
        labels[labels == idx] = it_labels
        lineage[k_max - 1] = labels

        k_max = k_max + 1
        
    return labels, lineage