# -*- coding: utf-8 -*-


# A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise
# Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu
# dbscan: density based spatial clustering of applications with noise
# https://raw.githubusercontent.com/choffstein/dbscan/master/dbscan/dbscan.py

UNCLASSIFIED = 'no-class'
NOISE = 'noise'


from collections import deque

# jarvis-patrick similarity, snn
def sim(p, q):
    return len(p & q)

def eps_neighborhood(p, q, eps):
    return sim(p, q) >= eps

def region_query(D, point_id, eps):
    n_points = len(D)
    neighborhood = set()

    # for i in xrange(n_points):
    for i in D[point_id]:
        if eps_neighborhood(D[point_id], D[i], eps):
            neighborhood.add(i)
        for j in D[i]:
            if eps_neighborhood(D[point_id], D[j], eps):
                neighborhood.add(j)
    return list(neighborhood)

def expand_cluster(D, assignment, point_id, cluster_id, eps, min_points):
    neighborhood = region_query(D, point_id, eps)
    if len(neighborhood) < min_points:
        assignment[point_id] = NOISE
        return False

    assignment[point_id] = cluster_id
    for seed_id in neighborhood:
        assignment[seed_id] = cluster_id

    q = deque(neighborhood)
    while len(q) > 0:
        q_point_id = q.popleft()
        neighborhood_q = region_query(D, q_point_id, eps)
        if len(neighborhood_q) < min_points:
            continue
    
        for j in neighborhood_q:
            if assignment[j] == UNCLASSIFIED:
                q.append(j)

            if assignment[j] == UNCLASSIFIED or assignment[j] == NOISE:
                assignment[j] = cluster_id
    return True   

def dbscan(D, eps, min_pts, verbose=0):
    cluster_id = 1
    n_points = len(D)
    assignment = [UNCLASSIFIED] * n_points

    for point_id in xrange(n_points):
        if verbose and point_id % 5000 == 0:
            print "iteration %d" % point_id

        if assignment[point_id] != UNCLASSIFIED:
            continue

        if expand_cluster(D, assignment, point_id, cluster_id, eps, min_pts):
            cluster_id = cluster_id + 1
        
    return assignment