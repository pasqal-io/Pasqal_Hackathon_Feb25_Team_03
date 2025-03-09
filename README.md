<h1 align="center"> Connected Cities </h1>
<h3 align="center">  City Planning in the Quantum Era</h3>
<h4 align="center"> Pasqal Challenge </h4>

<img style="padding-bottom:2em;max-width:50%;height:auto;display:block;margin-left:auto;margin-right:auto" alt="Hierarchical Cluster visualization with default values" src="imgs/example.png"/>


<div>
<font size="4">
<emph><strong>Authors</strong></emph>: Víctor Bayona Marchal, Jose Manuel Montes Armenteros, Yllari González Koda and Brian Sena Simons
</font>
</div>


# Table of contents
1. [Pasqal Challenge](#pasqal-Challenge)
   1. [Introduction](#introduction)
   2. [QUBO problem](#qubo-problem)
       1. [Conditions](#conditions)
   3. [Optimizations of weights](#optimization-of-weights)
   4. [Multi-route implementation](#multi-route-implementation)
   5. [VQAA: quantum component](#vqaa-quantum-component)
2. [Setup and Requirements](#setup-and-requirements)
    1. [Structure](#structure)
    2. [Requirements](#setup)
        1. [Linux](#linux)
        2. [Windows](#windows)
    3. [How-to](#how-to)
        1. [Visualize hierarchical cluster](#visualize-the-hierarchical-clustering-algorithm)
        2. [Download amenities from a city](#download-amenities-from-a-city)

# Pasqal Challenge

## Introduction
We aim to solve the problem of creating M bus lines, each with P bus stops in a city with a total number of N stops. We consider the following:

1. Distances are not symmetric (in general, $d(A,B) \neq d(B,A)$).
2. Bus stops are set based on social and demographic factors.
3. The number of bus stops per line, as well as the number of lines, are hyperparameters of the problem.

As the distances between stops are not symmetric, we can define the binary variables as

```math
    x_{ij}^{l} \in \{0,1\}
```

Where the value is 1 only if the line $l$ goes at some point from stop $i$ to stop $j$. The global function that we want to minimize is

```math
    f(x_{00}^{0}, x_{01}^{0},...,x_{0N-1}^{0}, x_{10}^{0}, ..., x_{N-1N-1}^{0}, x_{00}^{1}, ..., x_{N-1N-1}^{L-1}) = \sum_{l}^{L-1}\sum_{i}^{N-1}\sum_{j}^{N-1}D_{ij}x_{ij}^{l}
```

where the matrix of distances $D_{ij}$ is not symmetric and of order $N\times N$. The formulation of the optimization problem is based on the QUBO problem, i.e, minimizing

```math
F_Q(z) = z^TQz
```

where $z$ is a n-dimensional column vector and $Q$ an $n\times n$ matrix. To translate the problem to a QUBO formulation, we need some conditions and the binary variables.

## QUBO problem:

Using the formulation of the Traveling Salesman Problem, a single bus is implemented. An initial stop (i) and final stop (j) are selected, not necessarily being i=0, j=N-1. The route has p+1 stops. The distances remain asymmetric (but the routes will be understood to be bidirectional, therefore in this first implementation we symmetrized the distance cost matrix). The variables are now a total of $R=N(p+1)$ and are of the shape $(y_{0N+0}, ..., y_{0N+N-1}, y_{1N+0},...,y_{1N+N-1},... ,..., y_{pN+N-1})$. The encoding is now thought to be about time-steps. More information [here](https://github.com/microsoft/qio-samples/blob/danielstocker-slc-ship-loading/samples/traveling-salesperson/traveling-salesperson.ipynb).  In the proper Traveling Salesman formulation, the only difference is that $p=N$

The cost function is then

```math
    \sum_{k=0}^{p-1}\sum_{i=0}^{N-1}\sum_{j=0}^{N-1} (x_{Nk+i}\cdot x_{N(k+1)+j}D_{ij})
```

The cost function in the $Q_matrix$ representation is just inserting the distances cost matrix $D$ into blocks $0;1$, $1;2$,...,$N-2;N-1$. For symmetric purposes, the cost $Q_{matrix}^{T}$ is also added.

### Conditions: iteration 2

Every condition has a weight which is aimed to be optimized.

1. In each step the bus is in only one stop: $\sum_{k=0}^{p}\sum_{i=0}^{N-1}\sum_{j=0}^{N-1}(x_{i+Nk}x_{j+Nk})=0$ (In the reference this term is not symmetric as it takes only $i<j$. Here we consider the rest of the terms, which only adds more weight to the penalty).
2. In each step, the bus should be somewhere: $(\sum_{k}^{N(p+1)-1}x_{k}-(p+1))^2$
3. Just one visit to each node: $\sum_{k}^{N(p+1)-1}\sum_{f=k+N,step \;N}^{N(p+1)-1}x_{k}x_{f}=0$
4. Start node is negatively penalized: $-\lambda_4 x_{0+i}$
5. End node is negatively penalized: $-\lambda_5 x_{Np+j}$

### Optimization of weights

Some strategies for choosing the most suitable lambda parameters are followed. The first one is to fix all penalties to be the Upper Bound of the cost matrix, that is, the cost of a solution with all binary variables with the value 1. An optimization algorithm is implemented to reduce the high value this term imply.

### Multi-route implementation

An approach for finding new routes for the same stops which connect all the stops could be to select the new start and end nodes as well as eliminate the previous formed line's start and end nodes. Therefore, we allow for crossing lines but without starting or ending in the same nodes.

## VQAA: quantum component

The quantum component deployed in this work is a Variational Quantum Adiabatic Algorithm (VQAA). The idea is to use the QUBO formulation of the problem to embed it into Pasqal’s neutral atoms Hamiltonian. Once a register is created, we define a proper sequence to be run. This sequence includes the adiabatic pulse (which depends mainly on two parameters, amplitude and detuning) and the detuning map modulator, whose value depends on the detuning. The sequence is then run, and the parameters are updated to minimise the average cost of the sampled solutions.

# Setup and Requirements
## Structure
```bash
├── data
│   ├── amenities-granada.csv
│   ├── lamdasOptimized
│   ├── matriz-rutas-granada
│   ├── overpy-granada-query.txt
│   ├── Q_matrix_optimal_N_5_p_2_startNode_2_endNode_4
│   └── utils.py
├── imgs
│   ├── example.png
│   └── HierarchicalDivision.png
├── legacy_files
│   ├── analog_qaoa.ipynb
│   ├── data
│   ├── pulserQUBO.ipynb
│   ├── PulserQUBOMethods.py
│   ├── results
│   ├── TSP_QUBO_Solver.ipynb
│   └── TSP_QUBO_Solver_MULTILINE.ipynb
├── main
│   ├── pipe.py
│   ├── benchmark.ipynb
│   ├── tree
│   ├── tsp
│   └── vqaa
├── POC_classical.ipynb
├── POC.ipynb
├── README.md
└── requirements.txt

```
Regarding the directories, they are organized as follows:
- `docs`: Contains documentation about the project development and formulation.
- `data`: Contains the main python scripts that are related to data fetching and preprocessing.
- `main`: Contains the main python scripts to run the experiments.
    - `tree`: Contains the hierarchical clustering algorithm, `linkageTree` and visualization techniques `utils`.
    - `tsp`: Contains the adaptation of the travelman sales problem for bus line optimization.
    - `vqaa`: Contains the algorithms related to the quantum algorithm.
- `legacy_files`: Contains older files used in the development of the project which are no longer needed.
- `results`: Contains the result of the experiments.

Regarding the notebooks we have:
- `POC.ipynb`: Jupyter Notebook with the minimal working example for iteractive visualization of the proposed project.
- `POC_classical.ipynb`: Jupyter Notebook with an extension of the minimal working example using only classical solvers for visualization of a complete solution.


## Requirements
In order to run all the experiments make sure to install all the necessary libraries.
We recommend creating an environment to avoid incompatibility with other already installed python libraries in your system.
### Linux
```python
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```
### Windows
```python
python -m venv .venv
.\venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```
## How-to
### Visualize the hierarchical clustering algorithm.
We can simply run the algorithm in the main package as follows
```python
python -m main.tree.linkageTree -data data/amenities-granada.csv
```
Remember to pass a valid csv with the format similar to `overpy` library.
The output should be an image similar to the following:

<img style="max-width:50%;height:auto;display:block;margin-left:auto;margin-right:auto" alt="Hierarchical Cluster visualization with default values" src="imgs/HierarchicalDivision.png"/>

### Download amenities from a city.We can simply run the algorithm in the data package as follows
```python
python -m main.data.utils -output data/amenities-granada.csv -query data/overpy-granada-query.txt
```
We are using the `overpy` library to fetch the data. Therefore, the query file should follow overpy valid structure.
