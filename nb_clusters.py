#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 17:45:03 2018

@author: mira
"""

from spherecluster import SphericalKMeans
from scipy.sparse import load_npz
from time import time


#scp -i ~/Downloads/test.pem /home/mira/TAF/TER/code/*  ubuntu@ec2-18-188-138-62.us-east-2.compute.amazonaws.com:~/TER

def sphe_kmeans(matrix, n_clusters, nb_ex):
    labeler = SphericalKMeans(n_clusters = n_clusters, n_init=nb_ex,  max_iter=100)
    return labeler.fit(matrix)



def nb_clusters(matrix, nb_ex):
    file = open("/home/ubuntu/TER/code/resultat_aws.txt","a")
 
    nb_clust = 100
    start=time()
    sphe_km = sphe_kmeans(matrix, nb_clust, nb_ex)
#    min_inertia = sphe_km.inertia_
    print(sphe_km.inertia_)
    print(nb_clust , " : " , time()-start)
    
    file.write("\n" + str(nb_clust) + " , time : " + str(time()-start) + " , inertia : " + str(sphe_km.inertia_) + "\n")
     
    file.close() 
    
    choises = [200, 300, 400, 500]
    for i in choises:
        start=time()
        sphe_km = sphe_kmeans(matrix, i, nb_ex)
        print(sphe_km.inertia_)
        print(i , " : " , time()-start)
        file.write("\n" + str(i) + " , time : " + str(time()-start) + " , inertia : " + str(sphe_km.inertia_) + "\n")
     
        

if __name__ == "__main__":    
#    csr_meta = load_npz("/home/mira/TAF/TER/code/prod_term_matrix.npz")
    csr_meta = load_npz("/home/ubuntu/TER/code/prod_term_matrix.npz")
    
    nb_clusters(csr_meta, 5)
    
    #1/3, 100, 200, 300, 400, 500, 800, 1000
    
    
    