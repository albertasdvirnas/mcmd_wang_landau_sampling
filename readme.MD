This is Wang-Landau sampling algorithm for 2D Blume-Capel model with periodic boundary  conditions.

How to run the code:

To pre-generate possible states
python3.6  src/monte_carlo_wl.py 10 1000000 1
To compute the density of states
python3.6 src/monte_carlo_wl.py 10 1000000 2
To compute properties
python3.6 src/compute_specific_heat.py 8 1 1 example/

Possible extentions:
Magnetization could be computed at the last sten of density of states generation. Instead of maximum number of steps, a criteria for convergence could be included




References:
- SILVA, Cláudio José da; CAPARICA, A. A.; PLASCAK, João Antônio. Wang-landau monte carlo simulation of the blume-capel model. Physical Review E, 2006, 73.3: 036702.
- KWAK, Wooseop, et al. First-order phase transition and tricritical scaling behavior of the Blume-Capel model: A Wang-Landau sampling approach. Physical Review E, 2015, 92.2: 022134.
