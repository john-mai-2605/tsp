import sys
import numpy as np
import copy
import random
from .Prim_solver import good_path
def fitness(path):
        global num_fit 
        num_fit += 1
        assert len(path) == dim + 1
        score = 0
        for i in range(dim):
            score += dist[path[i+1]][path[i]]
        return score
class Network:
    def __init__(self, dist, dim):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.dim = dim
        self.visibility = 1/(dist + 1e-8)
        for i in range(dim):
            self.visibility[i, i] = 0 
            self.visibility[i, 0] = 0
        self.visibility /= np.amax(self.visibility)
        self.visibility_beta = self.visibility**beta
        self.pheromone = np.ones([dim, dim])
    def make_path(self):
        self.weights = np.multiply(self.pheromone**self.alpha, self.visibility_beta)
        weights = self.weights.copy()
        path = np.zeros(self.dim + 1, dtype = int)
        unvisited = [True for i in range(self.dim)]
        unvisited[0] = False
        for i in range(1, self.dim):  
            path[i] = random.choices(population = np.arange(self.dim)[unvisited], k = 1, weights = weights[path[i-1], unvisited])[0]
            unvisited[path[i]] = False
        if len(set(path)) != dim:
            print(set(path))
        assert len(set(path)) == dim
        return path
    def update(self, sol, best):
        for i in range(1, self.dim):
            deposit = best.score/sol.score
            self.pheromone[sol.path[i]][sol.path[i+1]] += deposit
            self.pheromone[sol.path[i+1]][sol.path[i]] = self.pheromone[sol.path[i]][sol.path[i+1]]
            # print(self.weights[5][18])
    def evaporate(self):
            self.pheromone *= (1-self.rho)

def mutation(path, dim):
    new_path = [path[i] for i in range(len(path))]
    i, j = np.random.choice(range(1, dim), 2, replace = False)
    if (i > j):
        i, j = j, i
    node1 = new_path[i]
    node2 = new_path[j]
    new_path.insert(i, node2)
    new_path.pop(j+1)
    return np.asarray(new_path)

class Solution:
    def __init__(self, path, dim):
        assert len(path) == dim + 1
        self.dim = dim
        self.path = path
        self.score = fitness(self.path)
    def mutate(self):
        new_path = mutation(self.path, self.dim)
        self.path = new_path
        self.score = fitness(self.path)
        return self
    def __str__(self):
        return f'path starts with {[node + 1 for node in self.path[:5]]} with score {str(self.score)}'

class Population:
    def __init__(self, network, size, dim, prim_path):
        self.size = size
        if not prim:
            self.pool = [Solution(network.make_path(), dim) for i in range(size)] 
        else:
            self.pool = [Solution(network.make_path(), dim) for i in range(size//2)] 
            self.pool += [Solution(prim_path, dim).mutate() for i in range(size//2, size)]
        self.dim = dim
        self.net = network
    def best(self):
        return min(self.pool, key = lambda x: x.score)

def ACO_solver(pop_size, max_fit, dim):
    net = Network(dist, dim)
    if prim:
        prim_path = good_path(dim, dist)
    else:
        prim_path = None
    population = Population(net, pop_size, dim, prim_path)
    global num_fit, history
    best = copy.deepcopy(population.best())
    while (num_fit <= max_fit):
        history.append([x.score for x in population.pool])
        if verbose:
            print(best)
        for ant in population.pool:
            if ant.score < best.score:
                best = copy.deepcopy(ant)
            net.update(ant, best)
            net.evaporate()
        net.update(best, best)            
        population = Population(net, pop_size, dim, prim_path)
        #print(net.pheromone)
    return population, best

