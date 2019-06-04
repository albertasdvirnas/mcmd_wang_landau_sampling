# Wang-Landau sampling

# Ising example
#  https://rajeshrinet.github.io/blog/2014/ising-model/
# Blume-Capel theory
#  SILVA, Cláudio José da; CAPARICA, A. A.; PLASCAK, João Antônio.
#  Wang-landau monte carlo simulation of the blume-capel model. Physical Review E, 2006, 73.3: 036702.
# and
# KWAK, Wooseop, et al.
# First-order phase transition and tricritical scaling behavior of the Blume-Capel model: A Wang-Landau sampling approach. Physical Review E, 2015, 92.2: 022134.

import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import logging
import os


# initial random state
def initial_state(N):
    # random spin configuration for initial configuration
    state = np.random.randint(3, size=(N,N))-1
    return state


#  parameters of configuration
def calc_parameters(lattice):
    # kinetic energy
    energy = 0
    # number of nonzero spins
    nS = 0
    N = len(lattice)
    for i in range(len(lattice)):
        for j in range(len(lattice)):
            S = lattice[i,j]
            nS += S**2
            #  periodic boundary conditions
            nb = lattice[(i+1)%N, j] + lattice[i,(j+1)%N]
            # nb = lattice[(i+1)%N, j] + lattice[i,(j+1)%N] + lattice[(i-1)%N, j] + lattice[i,(j-1)%N]
            energy += nb*S

    # divide by 4 because every neighbour is included 4 times
    return energy, nS


def random_move(logG, H, lattice, E, S, N, f):

    flipped = False

    # we had E, N, E - old energy, N - old num of states
    i = np.random.randint(0, N)
    j = np.random.randint(0, N)

    # new spin has to be different from old spin
    c = np.random.randint(3)-1

    # so if the same spin is drawn, we skip this step. Alternatively only draw from two numbers
    if c != lattice[i,j]:

        # neighbour energy contribution
        nb = (lattice[(i + 1) % N, j] + lattice[i, (j + 1) % N] + lattice[(i - 1) % N, j] + lattice[i, (j - 1) % N] )

        # new energy
        E_new = E + (c - lattice[i, j]) * nb

        # new number of nonzero spins
        S_new = S-lattice[i,j]**2+c**2

        # new and old gamma
        gamma_new = logG[np.int(np.round(E_new+2*N**2)), np.int(S_new)]
        gamma_old = logG[np.int(np.round(E+2*N**2)), np.int(S)]


        # accept the move using importance sampling
        rand = random.random()


        if rand < np.exp(gamma_old-gamma_new):
            lattice[i,j] = c
            E = E_new
            S = S_new
            flipped = True

        # update histogram at each step
        H[np.int(np.round(E + 2*N**2 )), np.int(S)] += 1.
        # update density of states at each step
        logG[np.int(np.round(E + 2*N**2)), np.int(S)] += f
    return logG, H, lattice, E, S, flipped



# this function simulates grid for given number of N, and saves the final result into a file
def simulate_grid(N , num_runs, f ):
    logG = np.zeros((4 * N ** 2 + 1, N ** 2 + 1))
    H = np.zeros((4 * N ** 2 + 1, N ** 2 + 1))
    lattice_st = initial_state(N)
    E, S = calc_parameters(lattice_st)
    numPts = []
    lattice = lattice_st
    for i in range(0, num_runs):
        logG, H, lattice, E, S, flipped = random_move(logG, H, lattice, E, S, N, f)

        if i % 100000 == 0 and i != 0:
            logging.warning('Saving the visited states for i='+str(i) )

            numPts.append(sum(sum(H > 0)))

    mat = H > 0
    np.savetxt('results/bc_'+str(N)+'x'+str(N)+'.txt', mat, fmt='%i', delimiter = ',')
    print(numPts)
    np.savetxt('results/bc_'+str(N)+'x'+str(N)+'_convergence.txt', numPts, fmt='%i', delimiter=',')
    print('In total found N = '+str(np.sum(mat))+' states')


def compute_wl(N,num_runs):
    logG = np.zeros((4*N**2+1,N**2+1))
    H = np.zeros((4*N**2+1,N**2+1))
    f = np.exp(0)
    flatness = 0.8


    # Actual grid saved in a txt file
    grid = np.loadtxt(open('results/bc_'+str(N)+'x'+str(N)+'.txt', "rb"), delimiter=",", skiprows=0)> 0

    # initial state
    lattice_st = initial_state(N)
    E, S = calc_parameters(lattice_st)

    density = []
    histogram = []
    idx = []
    fstep = []
    lattice = lattice_st
    for i in range(0,num_runs):
        logG, H, lattice, E, S,flipped = random_move(logG, H,lattice, E, S, N, f)
        if i % 100000 == 0:
            meanH = np.mean(H[grid])
            minH = np.min(H[grid])
            logging.warning('Running flatness check on i='+str(i) )
            logging.warning('f ' + str(f)+ ' i '  + str(i) +' minH ' +str(minH) + ' meanH' + str(meanH))

            print(f, i, minH, meanH * flatness)
            if minH > meanH*flatness:
                logging.warning('Flatness check is passed!')
                # curStep +=1
                z = np.copy(logG)
                # save log density of states
                density.append(z)
                # save the "flat" histogram
                histogram.append(H[:])
                # save index when flatness was satisfied
                idx.append(i)
                # save the step size
                fstep.append(f)
                H = np.zeros((4 * N ** 2 + 1, N ** 2 + 1))
                np.savetxt('results/bc_'+str(N)+'x'+str(N)+'f_'+str(f)+'.txt', logG, delimiter = ',')

                f = f/2

                # In case we start an independent run
                # lattice = initial_state(N)
                # E, S = calc_parameters(lattice)

                #  Example how to load saved txt file
                # my_matrix = np.loadtxt(open("matrix.txt", "rb"), delimiter=",", skiprows=0)

    # save final, only relevant when not a single success until the final step
    np.savetxt('results/bc_' + str(N)+'/bc_' + str(N) + 'x' + str(N) + 'f_'+str(f)+'_final.txt', logG, delimiter=',')
    # plt.matshow(logG/np.min(logG[grid]))




    # Surface plot example
    from mpl_toolkits.mplot3d import Axes3D
    # data = np.zeros((4*N**2+1,N**2+1))
    # data = logG
    # data[data==0] = False
    # x = range(len(logG[1]))
    # y = range(len(logG))
    # hf = plt.figure()
    # ha = hf.add_subplot(111, projection='3d')
    #
    # X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    # # ha.plot_surface(X, Y, data/3.)
    # ha.plot_wireframe(X, Y, data/3.)
    #
    # plt.show()


# initialization of parameters
print(sys.argv[0])
try:
    N = int(sys.argv[1])
    num_runs = int(sys.argv[2])
    method = int(sys.argv[3])

except:
    N = 4
    num_runs = 1000
    method = 1

try:
    os.makedirs('results/bc_'+str(N)+'/')
except:
    True
    # do nothing


filename = 'results/l.log'


logging.basicConfig(filename=filename, filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logging.warning('This will get logged to a file')

logging.warning('System parameters N '+str(N)+' and num_runs ' + str(num_runs))


if method ==1:
    f = 1
    simulate_grid(N, num_runs, f)
else:
    compute_wl(N,num_runs)

logging.warning('Computation finished successfully')
