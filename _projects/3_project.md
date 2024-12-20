---
layout: page
title: Trees are shy.
date: 2023-04-24 21:01:00
description: Simulating crown/canopy shyness in trees!
tags: Trees, Cellular automaton
img: assets/img/Eden_growths/CrownShyness_thumb.jpg
importance: 1
category: Fun
related_publications: false
bibliography: eden.bib

---
---

<div class="links text-center d-flex justify-content-center">
    <a href="https://github.com/Cup-cake-lover/Eden_cluster_growthtests.git" class="mx-2" style="color: #333;">
        <i class="fab fa-github fa-lg"></i> Code
    </a>
</div>
---

## Crown shyness

Crown shyness is a fascinating natural phenomenon observed in some tree species, where the uppermost branches of neighboring trees avoid touching each other. This creates distinct gaps in the forest canopy, often resembling intricate puzzle-like patterns when viewed from below. While the exact reasons for crown shyness are not fully understood, scientists believe it may help trees optimize light exposure, reduce the spread of harmful pests or diseases, and prevent mechanical damage from wind or friction. This harmonious spacing showcases the remarkable way in which trees coexist, balancing competition and cooperation within a shared environment.

<img src="https://upload.wikimedia.org/wikipedia/commons/5/5d/Dryobalanops_Aromatica_canopy.jpg" alt="Cool Banner GIF" style="width:100%; height:auto;">

During a project at the National Center for Biological Sciences (NCBS), Bengaluru, India, I explored the phenomenon of crown shyness from a biophysical perspective. Initially, our goal was to extract the boundaries of tree canopies and calculate the roughness exponent, $$\alpha$$. However, due to poor quality data (images of the canopies) and the limitations of our basic boundary extraction algorithm, the results were inconclusive. Despite that, the concept really intrigued me.

My interest in generating exclusion patterns stayed with me until I learned about Eden clusters, thanks to a senior professor I was in contact with at the time. You can find the project codes and example results, as usual, in my GitHub repository. [Eden_cluster_growthtests](https://github.com/Cup-cake-lover/Eden_cluster_growthtests.git)

## Eden Clusters

First proposed by [Murray Eden](https://api.semanticscholar.org/CorpusID:56348806)  as a model for tumor growth, Eden clusters have garnered significant interest in a variety of random growth processes. The concept is simple: start with a seed on a lattice and probabilistically add neighboring points. This iterative process results in the formation of a space-filling cluster. Over the years, numerous studies have been conducted on such models.

In this case, we simulate the growth of multiple Eden clusters on a lattice. Since Eden clusters tend to grow symmetrically with rough, uneven edges, they serve as a useful model for tree canopies. We assume that tree canopies cannot overlap, so as they grow, they exclude each other, creating the unique exclusion patterns we are looking to replicate.

### Assumptions

The Eden Growth model we are using is based on probabilistic cellular automata (CA). This means that a probabilistic set of rules governs how the structure grows within the lattice. For our model, we'll begin by listing the key assumptions:

- Trees grow, symmetrically but with an inherent stochasticity associated with it (irregular edges)
- Branches within trees won't form loops, ie, they won't intersect (Intra-tree exclusion)
- Two different trees are always differentiable, one tree understands the existence of the other, by a unique label.
- Trees 'sense' the other trees within an exclusion distance ($$r$$). And stops the spatial growth. (Inter-tree exclusion)

### The rules

Keeping these assumptions, A set of CA rules are constructed which are as follows. The simulations are done on a $$N \times N$$ square grid. Hence there are $$N^2$$ accessible cells. For an underlying model an Eden clustering algorithm is proposed. Basic rules are as follows,

- A cell is only reproducible/ creates an adjacent neighbour if it's 'Alive'.
- A cell can only grow into an 'idle' cell.
- One of the 8 neighbours (Moore neighbourhood) is chosen at random, and a new 'alive' cell is grown at a specific time instant which is attached to the previous neighbouring cell.

Since this model forms a space-filling cluster, an additional rule is added to form 'inter-branch' repulsion. This produces a cluster that doesn't produce loops. ([Deepak Dhar and Ramakrishna Ramaswamy](https://link.aps.org/doi/10.1103/PhysRevLett.54.1346))

- If an 'idle' cell has two or more neighbours 'alive' no growth happens to that cell.

Such a growth produces the following type of cluster, The diameter is $$D$$ is directly proportional to the grid dimensionality ($$d$$) as well as the number of new alive cells.

$$
D \propto N_{alive}^{1/d}
$$

Now to simulate multiple tree clusters on the same lattice and for each tree to sense each other, every tree node is uniquely labelled. And we propose a new model with two new rules added further.

- A generalized neumann neighbourhood is considered when a new cell is populated, this generalized Neumann neighbourhood is based on the exclusion distance ($$r$$) which can be controlled.

- While growing each cluster without forming loops, All new alive cell is adhered to follow these rules. 

<div class="row text-center">
    <div class="col-sm mt-2 mt-md-0">
        <img src="https://upload.wikimedia.org/wikipedia/commons/4/4d/Moore_neighborhood_with_cardinal_directions.svg" class="img-fluid" alt="Moore Neighborhood"/>
    </div>
    <div class="col-sm mt-2 mt-md-0">
        <img src="https://upload.wikimedia.org/wikipedia/commons/a/a1/Von_neumann_neighborhood.svg" class="img-fluid" alt="Von Neumann Neighborhood"/>
    </div>
</div>
<div class="caption">
    Moore Neighbourhood (Left) and Von Neumann Neighbourhood (Right)
</div>


With this set of rules, we can simulate our trees!

# Simulation

The simulation can be performed pretty easily with given set of rules. First we set up some helper functions to label the trees and to calculate the respective neighbourhood indices. All of them are then put into a single class `HelperFunctions`


{% highlight python %}
class HelperFunctions:
    def __init__(self, grid_size, repulsion_factor):
        self.grid_size = grid_size
        self.repulsion_factor = repulsion_factor
  
    def seed_parse(self, sample_array, filename):
        seeds = np.loadtxt(filename, delimiter=',')
        scatter_seeds = []
        for i, j in enumerate(seeds):
            [ii, ij] = [int(j[0]), int(j[1])]
            sample_array[tuple([ii, ij])] = i + 1
            scatter_seeds.append(j)
        return sample_array, np.array(scatter_seeds)

    def create_permutations(self, t_n, treeN):
        not_treeN = list(range(1, treeN))
        not_treeN.remove(t_n)
        return not_treeN

    def von_neumann_neighbours(self, temp_i, temp_j):
        r = self.repulsion_factor
        f = lambda s, r: {s} | {(a + x, b + d - x) for d in (1, -1) * r for a, b in f(s, r - 1) for x in (0, d)}
        return list(f((temp_i, temp_j), r))

    def moore_neighbours(self, arr, temp_i, temp_j):
        neighbours = [
            arr[temp_i - 1, temp_j], arr[temp_i + 1, temp_j],
            arr[temp_i, temp_j - 1], arr[temp_i, temp_j + 1],
            arr[temp_i - 1, temp_j - 1], arr[temp_i + 1, temp_j + 1],
            arr[temp_i + 1, temp_j - 1], arr[temp_i - 1, temp_j + 1]
        ]
        return neighbours
{% endhighlight %}

The function `create_permutaions` creates labels other than the current tree label. `von_neumann_neighbours` checks for a generalized Von neumann neighbourhood, and  `moore_neighbours` calculates a Moore neighbourhood.


Now we create a function to implement the rules mentioned above. To do this, we can write a very compact logic;

{% highlight python %}
from helper_functions import HelperFunctions
def tree(i, j, new_array, treeN, t_n, N, r, helper:HelperFunctions):
    p = 1 / 8
    arr = np.copy(new_array)
    not_treeN = helper.create_permutations(t_n, treeN)

    # Define all possible neighbor positions
    neighbors_pos = [
        (i - 1, j),     # Up
        (i, j - 1),     # Left
        (i - 1, j - 1), # Up-Left
        (i + 1, j),     # Down
        (i, j + 1),     # Right
        (i + 1, j + 1), # Down-Right
        (i + 1, j - 1), # Down-Left
        (i - 1, j + 1)  # Up-Right
    ]

    # Check if the current position contains the tree and is within the bounds
    if arr[i, j] == t_n and i > 0 and j > 0 and i < N - r - 1 and j < N - r - 1:

        # Loop through all neighboring positions
        for ni, nj in neighbors_pos:
            if arr[ni, nj] == 0 and p > np.random.uniform(0, 1):
                # Get the Moore and Von Neumann neighbors for this position
                neighbours = helper.moore_neighbours(arr, ni, nj)
                von_neighbours = helper.von_neumann_neighbours(ni, nj)

                # Apply the same logic to decide if this tree can grow into the neighbor
                if all(arr[tuple(m)] != k for m in von_neighbours for k in not_treeN) and sum(neighbours) <= 2 * t_n:
                    arr[ni, nj] = t_n

    return arr
{% endhighlight %}


Now I have written the code in such a way that I can import a file which contains seed value indices called `seeds.csv` to the directly start the simulation from that distribution. 

Finally to tie everything together we create a `run` file.

{% highlight python %}

import argparse
from tree_main import tree
from helper_functions import HelperFunctions
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib

#Set plotting style
matplotlib.rcParams['figure.dpi'] = 200
plt.style.use('seaborn-v0_8-colorblind')

#Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--r', type=int, help="Repulsion factor", required=True)
parser.add_argument('--t', type=int, help="Run time", required=True)
parser.add_argument('--g', type=int, help="Grid size", required=True)
parser.add_argument('--f', type=str, help="Seed file name", required=True)

args = parser.parse_args()

#Initialize variables
N = args.g
sample_array = np.zeros((N, N))
helper = HelperFunctions(grid_size=N, repulsion_factor=args.r)

#Parse seeds from the specified file (error handling for CSV)
try:
    new_arr, scatter_seeds = helper.seed_parse(sample_array, args.f)
except FileNotFoundError:
    print(f"Error: '{args.f}' file not found. Please provide the correct file.")
    exit()

#Set up the simulation parameters
t = 0
t_0 = args.t
treeN = int(len(scatter_seeds)) + 1
r = args.r

#Set up progress bar
pbar = tqdm(desc='Growing Trees Progress', total=t_0)

#Main simulation loop
while t < t_0:
    for i in range(N):
        for j in range(N):
            for t_n in range(1, treeN):
                new_arr = tree(i, j, new_arr, treeN, t_n, N, r, helper)
    t += 1
    pbar.update(1)

pbar.close()

#Create a binary array for visualization
binary_array = new_arr.copy()
binary_array[binary_array != 0] = 1

#Plotting the final result
plt.imshow(binary_array, cmap='Greens', origin='lower')
plt.scatter(scatter_seeds[:, 1], scatter_seeds[:, 0], marker='v', color='black', s=60)
plt.savefig('Shyness_picture.png')
plt.show()  # Optionally display the image immediately
{%endhighlight %}


This is then ran on a CLI. 

```python
python3 run.py --r 2 --g 200 --t 100 --f seeds.csv
```

Where the flags are as follows,

- `--r` flag : repulsion factor
- `--g` flag : Grid size 
- `--t` flag : final time till the simulation runs 
- `--f` flag : filename containing seed indices

After the simulation is finished, A final image will be opened in a seperate window. Which will look something like this

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Eden_growths/Shyness_picture.png" title="Example result" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Growth of Eden cluster with a specified seed distribution.
</div>



And ofcourse, we can create a random seed distribution generator and save them into a file to create a much more real scenario,

{% highlight python %}
parser = argparse.ArgumentParser()
parser.add_argument('--r', type=int,help="Exclusion factor",required=True)
parser.add_argument('--N', type=int,help="Number_of_trees",required=True)
parser.add_argument('--g', type=int,help="Grid size",required=True)
parser.add_argument('--s', type=int,help="Seed Value (Any random int)",required=True)

args = parser.parse_args()  

N = args.g
treeN = args.N
r = args.r
seed_value = args.s

def random_seed_lattice(N,r,treeN,seed_value):
  random.seed(seed_value)
  xs = random.sample(range(0, int(N-r-1)), treeN)
  ys = random.sample(range(0, int(N-r-1)), treeN)
  seed_list = np.array(list(zip(xs,ys)))
  print("These will be your tree nodes",seed_list)
  np.savetxt("seeds.csv",seed_list.astype(int),delimiter=',')


random_seed_lattice(N,r,treeN,seed_value)
{%endhighlight %}


And can be executed with similar flags as before to create an random distribution of the seeds.

```bash
python3 random_seed_gen.py --g 200 --r 2 --s 6969 --N 5
```

Finally, beautiful growth animations can also be created like this,

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Eden_growths/crownshyness.gif" title="Example result" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Animation of an example growth
</div>

## Random seed distribution 

The simulations can be ran with a random seed distribtuion on a much bigger grid with a more number of trees to create patterns which are more similar that are found in nature.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Eden_growths/Shyness_picture_2.png" title="Example result" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Growth with a random see distribution with 60 trees in 500x500 grid.
</div>


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Eden_growths/tree_growth_2.gif" title="Example result" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Growth with a random see distribution with 60 trees in 500x500 grid animated.
</div>



# Remarks

So, what did I learn from this? It's clear that such a simple model can't fully capture the intricate dynamics of crown shyness. However, it was an extremely engaging coding exercise, and more importantly, it allowed me to explore a fascinating natural process that might otherwise go unnoticed. The journey itself understanding and implementing the Eden Growth model was a rewarding experience, even if the results were limited in terms of real-world application.

### Acknowledgements

Sandra Elsa Sanjai, University of Padua, Italy.


Image credits:

<a href="https://commons.wikimedia.org/wiki/File:River_of_Blue.jpg">Dag Peak</a>, <a href="https://creativecommons.org/licenses/by/2.0">CC BY 2.0</a>, via Wikimedia Commons

<a href="https://commons.wikimedia.org/wiki/File:Dryobalanops_Aromatica_canopy.jpg">Patrice78500</a>, Public domain, via Wikimedia Commons

<a href="https://commons.wikimedia.org/wiki/File:Von_neumann_neighborhood.svg">MorningLemon</a>, <a href="https://creativecommons.org/licenses/by-sa/4.0">CC BY-SA 4.0</a>, via Wikimedia Commons

<a href="https://commons.wikimedia.org/wiki/File:Moore_neighborhood_with_cardinal_directions.svg">MorningLemon</a>, <a href="https://creativecommons.org/licenses/by-sa/4.0">CC BY-SA 4.0</a>, via Wikimedia Commons