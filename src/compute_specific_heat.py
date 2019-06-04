#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys
import glob, os
import re

def check_convergence(folder,N):


    grid = np.loadtxt(open(folder+'bc_' + str(N) + 'x' + str(N) + '.txt', "rb"), delimiter=",",
                      skiprows=0) > 0
    gridC = np.loadtxt(open(folder+'bc_' + str(N) + 'x' + str(N) + '_convergence.txt', "rb"), delimiter=",",
                      skiprows=0)
    try:
        x = np.linspace(100000,len(gridC)*100000,len(gridC))
        plt.figure()
        plt.plot(x,gridC)
        plt.xlabel('Number of MC steps')
        plt.ylabel('Number of visited states')
    except:
        print('Not enough data to print result')

    f = []
    logG = []
    for file in glob.glob(folder +'bc_'+str(N)+'/'+'bc*'):
        fname = file
        f.append(float(fname.split('_')[3].replace('.txt', '')))

        logG.append(np.loadtxt(open(fname, "rb"), delimiter=",", skiprows=0))

    args = np.argsort(f)[::-1]
    f.sort(reverse=True)

    print(f)
    nelts = np.sum(grid)
    err = []
    for ix in range(1,len(logG)):
        cur_en = logG[args[ix]][grid]
        prev_en= logG[args[ix-1]][grid]

        err.append(np.sum((np.array(cur_en)/nelts-np.array(prev_en)/nelts)**2))

    try:
        plt.figure()
        plt.plot(err,'x')
        plt.yscale('log')
        plt.ylabel('Sum of square errors')
        plt.xlabel('Iteration')
        plt.legend(['$\\frac{1}{|(E,S)|}\sum_{E,S}\log^2(\\frac{G_i(E,S)}{G_{i-1}(E,S)})$'])
        plt.show()
    except:
        print('Not enough data points to print sum of square errors')
    return logG[args[-1]],grid

def compute_specific_heat(N,J,D,T,folder):
    # load energy and logG, both are matrices


    logG, grid = check_convergence(folder, N)

    # normalization
    #  like in example, lgC = log( (exp(lngE[0])+exp(lngE[-1]))/4. ), but making sure that we're not overflowing
    if logG[0][-1] < logG[-1][-1]:
        lgC = logG[-1][-1] + np.log(1 + np.exp(logG[0][-1] - logG[-1][-1])) - np.log(4.)
    else:
        lgC = logG[0][-1] + np.log(1 + np.exp(logG[-1][-1] - logG[0][-1])) - np.log(4.)
    logG -= lgC
    logG[logG < 0] = 0
    plt.matshow(logG)

    # logG = logG/np.sum(logG)

    J = 1.
    D = 1.5

    energy = [0]*T
    energy2 = [0]*T
    denE = [0]*T

    for i in range(0,len(logG)):
        for S in range(0,len(logG[0])):
            if grid[i,S]:
                E = i- 2*N**2
                energy += [(-J*E+D*S)*np.exp(logG[i,S]+(E*J-D*S)/t) for t in T]
                denE += [np.exp(logG[i,S]+(E*J-D*S)/t)for t in T]
                energy2 += [(-J*E+D*S)**2*np.exp(logG[i,S]+(E*J-D*S)/t) for t in T]


    C = (energy2/denE - (energy/denE) ** 2) / (T ** 2)
    plt.figure()
    plt.plot(T,C,label='D='+str(D)+' , '+'J='+str(J))
    plt.ylabel('$C_{J,D}(T)$')
    plt.xlabel('T')
    plt.legend()


    plt.figure()
    plt.plot(T,energy/denE,label='D='+str(D)+' , '+'J='+str(J))
    plt.ylabel('Internal energy <E>')
    plt.xlabel('T')
    plt.legend()
    plt.show()

    return C


# initialization of parameters
print(sys.argv[0])
try:
    N = int(sys.argv[1])
    J = int(sys.argv[2])
    D = int(sys.argv[3])
    folder = sys.argv[4]

except:
    N = 8
    J = 1.
    D = 1.
    folder = 'example/'+'bc_'+str(N)+'/try1/'



T = np.linspace(0.1, 5, 40)


C = compute_specific_heat(N,J,D,T,folder)


# Finallym we calculate distribution of a new variable, E1-sE2, with s - field mixing parameter
