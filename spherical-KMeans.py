#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 10:04:27 2018

@author: mira
"""

from spherecluster import SphericalKMeans
from scipy.sparse import load_npz
from time import time
import pickle

#scp -i ~/Downloads/test.pem /home/mira/TAF/TER/code/*  ubuntu@ec2-18-188-138-62.us-east-2.compute.amazonaws.com:~/TER

def sphe_kmeans(matrix, n_clusters, nb_ex):
    labeler = SphericalKMeans(n_clusters = n_clusters, n_init=nb_ex,  max_iter=100)
    return labeler.fit(matrix)

if __name__ == "__main__":    
#    csr_meta = load_npz("/home/mira/TAF/TER/code/prod_term_matrix.npz")
    
    ###########################################################################
    #                               PROD x TERM
    ###########################################################################
    
    csr_term = load_npz("/home/ubuntu/TER/code/prod_term_matrix.npz")
    start = time()
    sphe_km_prod_term = sphe_kmeans(csr_term, 200, 5)
    file = open("/home/ubuntu/TER/code/.git/spheKM_prod_term.txt","a")
    file.write("\n" + "200 clusters , time : " + str(time()-start) + " , inertia : " + str(sphe_km_prod_term.inertia_) + "\n")
    file.close() 
    
    
    with open(".git/spheKM_prod_term.pkl", 'wb') as file:  
        pickle.dump(sphe_km_prod_term, file)
        
    ###########################################################################
    #                               PROD x USER
    ###########################################################################
    
    csr_user = load_npz("/home/ubuntu/TER/code/prod_user_matrix.npz")
    start = time()
    sphe_km_prod_user = sphe_kmeans(csr_user, 200, 5)
    file = open("/home/ubuntu/TER/code/.git/spheKM_prod_user.txt","a")
    file.write("\n" + "200 clusters , time : " + str(time()-start) + " , inertia : " + str(sphe_km_prod_term.inertia_) + "\n")
    file.close() 
    
    with open(".git/spheKM_prod_user.pkl", 'wb') as file:  
        pickle.dump(sphe_km_prod_user, file)
        
    
    ###########################################################################
    #                               PROD x PROD (comp)
    ###########################################################################
    
    csr_comp = load_npz("/home/ubuntu/TER/code/csr_comp.npz")
    start = time()
    sphe_km_comp = sphe_kmeans(csr_comp, 200, 5)
    file = open("/home/ubuntu/TER/code/.git/spheKM_comp.txt","a")
    file.write("\n" + "200 clusters , time : " + str(time()-start) + " , inertia : " + str(sphe_km_prod_term.inertia_) + "\n")
    file.close() 
    
    with open(".git/spheKM_comp.pkl", 'wb') as file:  
        pickle.dump(sphe_km_comp, file)
        
        
    ###########################################################################
    #                               PROD x PROD (sub)
    ###########################################################################
    
    csr_sub = load_npz("/home/ubuntu/TER/code/csr_sub.npz")
    start = time()
    sphe_km_sub = sphe_kmeans(csr_sub, 200, 5)
    file = open("/home/ubuntu/TER/code/.git/spheKM_sub.txt","a")
    file.write("\n" + "200 clusters , time : " + str(time()-start) + " , inertia : " + str(sphe_km_prod_term.inertia_) + "\n")
    file.close() 
    
    with open(".git/spheKM_sub.pkl", 'wb') as file:  
        pickle.dump(sphe_km_sub, file)
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
