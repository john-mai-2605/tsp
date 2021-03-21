import math
import random
import numpy as np
from sklearn.metrics import pairwise_distances
import sys
from IO import get_input
from .Prim_solver import good_path
# dim, dist = get_input('a280.tsp')
# num_fit = 0

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
    path = np.zeros(length + 1, dtype = int)
    path[1:-1] = np.random.choice(range(1, length), size = length-1, replace = False) 
    return path

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

# Gene operation
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

def crossover(p1, p2, dim):
    mask = np.ones(dim + 1, dtype = int)  
    mask[1:-1] = [np.random.randint(2) for i in range(dim - 1)]
    path1 = [p1[i] if mask[i] else None for i in range(dim + 1)]
    path2 = [p2[i] if mask[i] else None for i in range(dim + 1)]
    inpath1 = np.zeros(dim + 1, dtype = int)  
    inpath2 = np.zeros(dim + 1, dtype = int)  
    for i in range(dim+1):
        if path1[i] is not None:
            inpath1[path1[i]] = 1
        if path2[i] is not None:
            inpath2[path2[i]] = 1
    for i in range(dim+1):
        if path1[i] is None and inpath1[p2[i]] == 0:
            path1[i] = p2[i]
            inpath1[p2[i]] = 1
        if path2[i] is None and inpath2[p1[i]] == 0:
            path2[i] = p1[i]
            inpath2[p1[i]] = 1
    remainder1 = [i for i in range(dim) if inpath1[i] == 0]
    remainder2 = [i for i in range(dim) if inpath2[i] == 0] 
    for i in range(dim+1):
        if path1[i] is None:
            path1[i] = remainder1.pop(-1)
        if path2[i] is None:
            path2[i] = remainder2.pop(-1)
    assert len(set(path1)) == dim
    assert len(set(path2)) == dim
    return np.asarray(path1), np.asarray(path2)

def selection(pool, dim, k, group_size = 5):
    chosen_sol = []
    for i in range(k):
        chosen_group = random.sample(pool[1:-1], group_size)
        chosen_sol.append(chosen_group[np.argmin([x.score for x in chosen_group])])
    return chosen_sol

# Solution class
class Solution:
    def __init__(self, path, dim, scoring = False):
        assert len(path) == dim + 1
        self.dim = dim
        self.path = path
        if scoring:
            self.score = fitness(self.path)
    def update(self):
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
            self.pool = [Solution(random_path(dim), dim, True) for i in range(size)] 
        else:
            prim_path = good_path(dim, dist)
            self.pool = [Solution(random_path(dim), dim, True) for i in range(size//2)] 
            self.pool += [Solution(prim_path, dim, True).mutate() for i in range(size//2, size)]
        self.dim = dim
    def best(self):
        return min(self.pool, key = lambda x: x.score)
    def select(self, k, group_size = 5):
        return selection(self.pool, self.dim, k, group_size)
    def cross(self, parent1, parent2):
        return [Solution(path, self.dim) for path in crossover(parent1.path, parent2.path, self.dim)]
    def evolution(self, offspring):
        self.pool.extend(offspring)
        sorted_pool = sorted(self.pool, key = lambda x: x.score)
        chosen_from_best = int(self.size*pressure)
        chosen_from_remaining = self.size - chosen_from_best
        self.pool = sorted_pool[: chosen_from_best] + random.sample(sorted_pool[self.size//2:], chosen_from_remaining)
        assert len(self.pool) == self.size
        # self.pool = self.select(self.size)
def GA_solver(pop_size, max_fit, dim):
    population = Population(pop_size, dim)
    global num_fit, history
    while (num_fit <= max_fit):
        history.append([x.score for x in population.pool])
        offspring = []
        while len(offspring) < pop_size:
            parent1, parent2 = population.select(2, 5)
            child1, child2 = population.cross(parent1, parent2)
            child1.mutate()
            child2.mutate()
            offspring.extend([child1, child2])
        population.evolution(offspring)
        if verbose:
            print(population.best())
    return population, population.best()
