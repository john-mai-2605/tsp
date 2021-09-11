Some simple algorithms attempting to solve TSP - see report.pdf for more details
## Code documentation
### 1. Structure and how to run:
The solver implements 3 algorithm: Genetic Algorithm, Particle Swarm Optimization and Ant Colony Optimization

To run the code:

**Step 1:** Open the terminal for tsp/ directory

**Step 2:** Put the .tsp file in the same directory

(I include burma14.tsp, berlin52.tsp, att48.tsp, ch130.tsp and a280.tsp if you want to test immediately. For other files, please put it yourself)

**Step 3:** Install dependencies

pip install -r requirement.txt

**Step 4:** Run

Refer to the runner interface (3).

Example: The command to run burma14.tsp with default parameters is

python solver.py burma14.tsp

### 2. Dependencies: NumPy, matplotlib, scikit-learn

pip install -r requirement.txt

### 3. Runner interface:
python solver.py source [-p P] [-f F] [-solver SOLVER] [-graph GRAPH]
[-prim PRIM] [-pressure PRESSURE] [-w W] [-c1 C1] [-c2 C2]
[-a A] [-b B] [-r R]

Positional arguments: source Input file

Optional arguments:

|   Arg   |              Meaning                |   Default   |   Type   |
|---------|:-----------------------------------:|:-----------:|:---------|
|  -p P   |POPULATION                           |   10        |  int     |
|-f F     |MAX FITNESS CALL                     |   100000    |  int     |
|-solver  | SOLVER |solver (algorithm to use) |ACO| str|
|-graph   |GRAPH |Draw |graph| or| not |False| bool|
|-prim | PRIM | Initialize using Prim algorithm or not |False |bool|
|-pressure| PRESSURE| GA selection pressure | 0.5 | float|
|-w W -c1 C1 -c2 C2 | PSO weights| 4 2 1| float|
|-a A -b B -r R |ACO weights (alpha, beta, rho) |1 3 0.1 |float|
|-v V |Verbosity |False| bool|

Example: To solve burma14.tsp with GA, show final graph and all other default settings

python solver.py burma14.tsp -solver GA -graph True
