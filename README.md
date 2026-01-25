This repository contains the code used to generate the results in **"Percolation Critical Probability of Aperiodic Smith Hat"** by Haitao Gao and Aaryash Bharadwaj.

## Overview

The code performs Monte Carlo simulations to determine the percolation critical probability (p_c) for both site and bond percolation on the Hat aperiodic monotile. 
The analysis uses finite-size scaling to extrapolate critical probabilities to the infinite lattice limit.

## Code Structure

The program is split into the following components: 

1. hat_tiling: The code uses a slightly modified version of the Aperiodic Tile generator found alongside the original paper on "An Aperiodic Smith Monotile"
   (Smith, Myers, Kaplan, and Goodman-Strauss, 2023) by the University of Waterloo.
2. graph_builder: Converts the structure into a graph which can be used for both bond and union percolation. It also creates the square-frame used for
   Monte-Carlo simulation.
3. The percolation engine. Checks left-right percolation and up-down percolation and finding percolation based on the union and intersection of these. In the
   limit, a correct percolation would have intersection and union converge to a single value as the square-frame grows larger. Something we use to find the
   threshold. The  percolation uses union-find to find if two nodes are in the same component.
4. Visualiser: Displays results including the critical value as L increases, the finite-scaling for both site and bond as well as the patch itself and the regions tested.
5. Main: The primary execution. Has various fine-tunable parameters to increase speed, precision or range within experimentation.

Note: The final result is the patch with the square-frame regions shown. Whilst potentially useful (hence inclusion) is not a result and can be slow to generate.

## Requirements
```
numpy
scipy
matplotlib
tqdm
```
### Parameters
- `--r`: Patch recursion depth (default: 5, controls tiling size). 
- `--t`: Number of Monte Carlo trials per system size (default: 1000)
- `--Lmin`: Minimum lattice size (default: 10.0)
- `--Lmax`: Maximum lattice size (default: 400.0)
- `--Lstep`: Step size between lattice sizes (default: 10.0)
- `--bt`: Boundary thickness for edge detection (default: 1.0)

Note: Changing --r may also require changing analyze_square_frame() within graph_builder.py, we thus recommend not doing so for simple usage.

### Examples

For a quick test (Estimate 5 minutes or less):
```bash
python main.py --t 100 --Lmax 100
```

For high-precision results (warning: 20+ hours runtime):
```bash
python main.py --t 1000 --Lmax 400
