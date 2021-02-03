#!/usr/bin/python
from __future__ import print_function
import sys
import subprocess
import os
from itertools import *
import math as M
import numpy as np
import matplotlib.pyplot as plt

#Defining the arguments or file input
if len(sys.argv) == 2:
    qcin = sys.argv[1]
    qc_base = qcin.split('.')[0]
else:
    print ('Incorrect filetype')
    sys.exit()

#Defining important functions

def get_contents(filename):
    """
    Function to read the content of a file
    """
    with open(filename, 'r') as f:
        contents = f.readlines()
    return contents

    
def All_Coordinates(qcin):
    """
    A comprehensive function to pull out XYZ coordinates from an XYZ trajectory file
    """
    qc_input = get_contents(qcin)
    temp =[]
    Coords =[]
   
    for line in qc_input:
        data = line.split()
        if len(data) >= 4:
            temp.append([float(i) for i in data[1:4]])
        else:
            if len(temp) !=0:
                Coords.append(temp)
            temp =[]
    return  Coords
    
    
def Align_to_Ref(Ref, Coords):
    """
    Function that aligns a coordinate to a reference in order to minimise the RMSD
    """

    
    #find the center of each coordinates
    center_0 = np.mean(Ref, axis=0)
    center_1 = np.mean(Coords, axis=0)
    
    #Center each coordinate
    centered_ref = Ref - center_0
    coord_to_com = Coords - center_1
    
    # Find the rotation that will align coord to Ref
    #Computation of the covariance matrix
    M = np.dot(np.transpose(coord_to_com),centered_ref)
    #M = coord_1.transpose().dot(coord_0)
    
    #computing the SVD of the covariance matrix
    U,S, V_H = np.linalg.svd(M)
    
    #Decide whether we need to correct our rotation matrix to ensure a right-handed coordinate system
    d = (np.linalg.det(U) * np.linalg.det(V_H)) < 0.0
    if d:
        S[-1] = -S[-1]
        U[:, -1] = -U[:, -1]
        
    #Compute rotation matrix
    Rotation_Matrix = np.dot(U, V_H)
    # Align the two sets of coordinates and calculate the RMSD
    aligned_coords = coord_to_com.dot(Rotation_Matrix)
    
    return aligned_coords, centered_ref
    
def RMSD(Ref, Coords):
    """
    Function that computes the RMSD between two coordinates
    """
    aligned_coords, centered_ref = Align_to_Ref(Ref, Coords)
    
    distance = list(map(lambda i, j :np.linalg.norm(i-j)**2 ,aligned_coords, centered_ref))
    
    rmsd = M.sqrt(sum(distance)/float(len(centered_ref)))
    return rmsd

#Main part of the program

#Read trajectory coordinates
Coordinates= All_Coordinates(qcin)

#computing pairwise RMSD
rmsd= starmap(RMSD, product(Coordinates, Coordinates))

#reshaping 1D list into numpy matrix for easy density plot
distance= np.array(list(rmsd)).reshape(len(Coordinates), len(Coordinates))

#Plotting pairwise RMSD
ticks =np.arange(0, len(distance)+1, 1000)
label = np.arange(0, len(ticks), 1)

plt.imshow(distance , cmap='viridis', origin='lower')
plt.xlabel(r'Time (ps)')
plt.ylabel(r'Time (ps)')
plt.xticks(ticks, label)
plt.yticks(ticks, label)
plt.colorbar(label=r'RMSD ($\AA$)')
plt.savefig('RMSD.png',dpi=2000, bbox_inches = 'tight')
#plt.show()


