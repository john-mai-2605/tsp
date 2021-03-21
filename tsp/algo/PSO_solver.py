import math
import random
import numpy as np
from sklearn.metrics import pairwise_distances
import sys
from IO import get_input
from .Prim_solver import good_path


# Fitness function:
def fitness(path):
    global num_fit 
    num_fit += 1
    assert len(path) == dim + 1
    score = 0
    for i in range(dim):
        score += dist[path[i+1]][path[i]]
    return score


# Random solution
def random_path(length):
    weights = 1/(dist + 1e-8)
    path = np.zeros(length + 1, dtype = int)
    for i in range(length):
        weights[i, i] = 0 
    weights[:, 0] = 0

    unvisited = [True for i in range(length)]
    unvisited[0] = False
    for i in range(1, length):  
        path[i] = random.choices(population = np.arange(length)[unvisited], k = 1, weights = weights[path[i-1], unvisited])[0]
        unvisited[path[i]] = False
    assert len(set(path)) == dim
    return path

def relu(x):
    x = np.clip(x, 0, None)
    x = x / np.linalg.norm(x, ord = 1)
    return x

# Solution class
class Solution:
    def __init__(self, dim, scoring = False, init = None):
        self.dim = dim
        if init is not None:
            self.position =  init + abs(np.random.normal(0, 0.1, [dim, dim]))
        else:
            self.position = np.random.rand(dim, dim)
        for i in range(dim):
            self.position[i, i] = self.position[i, 0] = 0
        self.position = relu(self.position)
        self.velocity = np.zeros([dim, dim])
        self.decode()
        if scoring:
            self.score = fitness(self.path)
        self.pbest = copy.deepcopy(self)

    def decode(self):
        self.path = np.zeros(self.dim + 1, dtype = int)
        unvisited = [True for i in range(self.dim)]
        unvisited[0] = False
        for i in range(1, self.dim):  
            available = np.arange(self.dim)[unvisited]
            self.path[i] = available[np.argmax(self.position[self.path[i-1], unvisited])]
            unvisited[self.path[i]] = False
        assert len(set(self.path)) == self.dim
    
    def move(self, gbest):
        self.velocity = w*self.velocity + c1*random.random()*(self.pbest.position - self.position) + c2*random.random()*(gbest.position - self.position)
        self.position += self.velocity
        for i in range(self.dim):
            self.position[i, i] = self.position[i, 0] = 0  
        self.position = relu(self.position)
        self.decode() 
        self.score = fitness(self.path)  

    def mutate(self):
        new_path = mutation(self.path, self.dim)
        self.path = new_path
        self.update()
        return self

    def __str__(self):
        return f'path starts with {[node + 1 for node in self.path[:5]]} with score {str(self.score)}'

# Population class:
class Population:
    def __init__(self, size, dim):
        self.size = size
        if not prim:
            self.pool = [Solution(dim, True) for i in range(size)] 
        else:
            self.pool = [Solution(dim, True) for i in range(size // 2)] 
            prim_path = good_path(dim, dist)
            init = np.zeros([dim, dim])
            for i in range(dim):
                init[prim_path[i], prim_path[i+1]] = 1
            self.pool += [Solution(dim, True, init) for i in range(size//2, size)]
        self.dim = dim
    def best(self):
        return min(self.pool, key = lambda x: x.score)
import copy
def PSO_solver(pop_size, max_fit, dim):
    population = Population(pop_size, dim)
    global num_fit
    global history
    gbest = copy.deepcopy(population.best())
    while (num_fit <= max_fit):
        if verbose:
            print(gbest) 
        history.append([x.score for x in population.pool])
        for sol in population.pool:
            sol.move(gbest)
            if sol.score < sol.pbest.score:
                sol.pbest = copy.deepcopy(sol)
            if sol.score < gbest.score:
                gbest = copy.deepcopy(sol)
    return population, gbest
