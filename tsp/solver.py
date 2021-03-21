import math
import random
import numpy as np
from sklearn.metrics import pairwise_distances
import sys
from IO import get_input, export_graph, export_csv
from algo import GA_solver, PSO_solver, ACO_solver
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('source', type=str, help="Input file")
parser.add_argument('-p', type=int, default='10', help="POPULATION")
parser.add_argument('-f', type=int, default='100000', help="MAX FITNESS CALL")
parser.add_argument('-solver', type=str, default='ACO', help="solver")
parser.add_argument('-graph', type=bool, default=False, help="Draw graph or not")
parser.add_argument('-prim', type=bool, default=False, help="Initialize using Prim algorithm or not")
parser.add_argument('-pressure', type=float, default=0.5, help="GA selection pressure")
parser.add_argument('-w', type=float, default=0.5, help="PSO weights")
parser.add_argument('-c1', type=float, default=4, help="PSO weights")
parser.add_argument('-c2', type=float, default=2, help="PSO weights")
parser.add_argument('-a', type=float, default=1, help="ACO weights")
parser.add_argument('-b', type=float, default=3, help="ACO weights")
parser.add_argument('-r', type=float, default=0.1, help="ACO weights")
parser.add_argument('-v', type=bool, default=False, help="Verbosity")

args = parser.parse_args()
source = args.source
if args.solver == 'GA':
	solver = GA_solver
	runner = solver.GA_solver
	solver.pressure = args.pressure
elif args.solver == 'PSO':
	solver = PSO_solver
	runner = solver.PSO_solver
	solver.w = args.w
	solver.c1 = args.c1
	solver.c2 = args.c2
elif args.solver == 'ACO':
	solver = ACO_solver
	runner = solver.ACO_solver
	solver.alpha = args.a
	solver.beta = args.b
	solver.rho = args.r
else:
	raise NotImplementedError

solver.prim = args.prim
solver.verbose = args.v
solver.dim, solver.dist, coors = get_input(source)
solver.num_fit = 0
solver.history = []
POPULATION = args.p
MAX_FIT = args.f
population, best = runner(POPULATION, MAX_FIT, solver.dim)
print(best.score)
export_csv(best)
if args.graph:
	export_graph(solver.history, best, coors)