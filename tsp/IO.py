import math
import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

def get_input(file = 'a280.tsp'):
	
	# Open input file
	with open(file, 'r') as f:
	    content = f.read()

	# Read file
	lines = content.splitlines()
	lines = [x.lstrip() for x in lines if x != ""]
	for line in lines:
		if line.startswith('DIMENSION'):
			dim = int(line.split(': ')[1])
			break
	for i, line in enumerate(lines):
		if line.startswith('1'):
			break
	data = lines[i:i+dim]
	assert len(data) == dim

	# Extract coordinate
	coors = []
	for datum in data:
	    x, y, z = datum.strip().split()
	    coors.append((int(x), float(y), float(z)))
	coors = np.asarray(coors)

	# Distance matrix
	dist = pairwise_distances(coors[:, 1:], coors[:, 1:])
	dist = np.asarray(dist)
	return dim, dist, coors

def export_csv(best):
	content = [str(city+1) for city in best.path[:-1].tolist()]
	content = '\n'.join(content)
	with open('solution.csv', 'w') as f:
		f.write(content)

def export_graph(history, best, coors):
	history = np.asarray(history)
	fig, ax = plt.subplots(1, 2)
	ax[0].plot(range(len(history)), history, '.', color='green')
	ax[0].plot(range(len(history)), np.min(history, axis = 1), '-', color='blue')

	best_path = best.path
	points = coors[best_path, :]
	ax[1].plot(points[:, 1], points[:, 2], 'o-b')

	plt.show()