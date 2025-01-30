# Pasqal Challenge

## Introduction 
We aim to solve the problem of creating M bus lines each with P bus stops in a city. The formulation is based on the QUBO problem, i.e, minimizing 
$$
F_Q(X) = X^TQX
$$

where $X$ is a n-dimensional column vector and $Q$ an $n\times n$ matrix. In our context, $X$ will represent the adjancency matrix and Q the cost function (distance). As we want to mninimize the distance from all bus lines, we perform summation over the M lines
$$
f_q(X) = \sum_l^M(X^l)^TQX^l
$$
where $l$ is each for each line.

## Conditions
Every condition is weighted according to it's importance.

1. Every bus stop has atleast one exit: $\sum_{l=0}^{L-1}\sum_{i=0}^{N-1} x_{ij}^{l} \geq 1 \forall j $ 
2. Every bus stop has atleast one entrance: $\sum_{l=0}^{L-1}\sum_{j=0}^{N-1} x_{ij}^{l} \geq 1 \forall i $
3. Every bus stop goes to a different bus stop: $x_{ii}^{l} = 0 \forall i, l$ 
4. If a bus stops at A it must then leave from A: $\textrm{if } x_{ij}^{l} = 1 \textrm{ then } \sum_{k=0}^{N-1} x_{jk}^{l} = 1 \forall i, j, l$
5. Every line should be closed: $\sum_l^L\sum_i^N\sum_j^N x_{ij}^{l} = N$
    1. Every line should be covered: [placeholder]
6. There should be a closed path that goes through each node 
    1. Every node should be in a accesible cicle


## Code Description

- `get_bus_stops.ipynb`: Jupyter Notebook where we view and generate the amenities from a city. We can also visualize the result of QUBOSolver.
- `graph_checking.ipynb`: Jupyter Notebook with POC's for matrix computations.
- `QUBOSolver.ipynb`: Jupyter Notebook with the QUBOSolver POC that generates the solution for a given graph.
- `busLinesQUBOMethods.py`: Where the QUBO creation and constraints are written. 
- `data/`: Where input files are stored.
- `results/`: Where output files are stored.