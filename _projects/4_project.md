---
layout: page
title: Learning from the Ising model!  
date: 2023-04-24 21:01:00
description: Boltzmann machine learning!
tags: Ising model, Boltzmann learning
img: assets/img/Boltzmann_plots/Ising_quench_thumb.gif
importance: 1
category: Machine learning
related_publications: false
bibliography: 

---
## Boltzmann machine learning

Boltzmann machines are deeply connected to concepts from statistical mechanics, where equilibrium distributions and energy landscapes play central roles. By leveraging these connections, Boltzmann machines can model complex distributions, offering insights into both machine learning and physical systems. To understand the foundation of Boltzmann learning, we first explore the Ising model, a cornerstone in statistical physics. 

---

## The Ising Model

The Ising model provides a framework to study the collective behavior of spins arranged on a lattice. Each spin, $$ s_i $$, can take one of two values, $$ +1 $$ (up) or $$ -1 $$ (down). Spins interact with their neighbors via pairwise couplings $$ J_{ij} $$, and the system may also be influenced by external fields $$ h_i $$. The Hamiltonian for a generalized spin glass system is:

$$
\mathcal{H}(\{s_i\}) = -\sum_{j > i} J_{ij} s_i s_j - \sum_{i} h_i s_i
$$

Here:  
- $$J_{ij}$$: Interaction strength between spins $$ s_i $$ and $$ s_j $$.  
- $$h_i$$: External field acting on spin $$ s_i $$.  

This energy-based model encapsulates the interplay of spin alignments and external influences.

---

## Boltzmann Distribution and Partition Function

The Ising system's equilibrium probability distribution is given by the Boltzmann distribution:

$$
P_{J_{ij}, h_i}(\{s_i\}) = \frac{e^{-\mathcal{H}(\{s_i\})}}{\mathcal{Z}}
$$

where the partition function, $$ \mathcal{Z} $$, ensures normalization:

$$
\mathcal{Z} = \sum_{\{s_i\}} e^{-\mathcal{H}(\{s_i\})}
$$

This distribution forms the basis for statistical inference, as it describes the likelihood of any spin configuration $$ \{s_i\} $$.

---

## Kullback-Leibler (KL) Divergence

To measure the difference between the model distribution $$ P_\theta(S) $$ (parameterized by $$ \theta = \{J_{ij}, h_i\} $$) and the empirical distribution $$ P_0(S) $$, we use the Kullback-Leibler divergence:

$$
D_{KL}(P_0 || P_\theta) = \sum_{\{s_i\}} P_0(\{s_i\}) \log \left( \frac{P_0(\{s_i\})}{P_\theta(\{s_i\})} \right)
$$

Minimizing the KL divergence aligns the model distribution with the empirical data distribution.

---

## Deriving the Loss Function

The negative log-likelihood loss function measures how well the model $$ P_\theta(\{s_i\}) $$ approximates the empirical distribution $$ P_0(\{s_i\}) $$. It is derived as follows:

The likelihood of the observed data under the model is:

$$
\mathcal{L}(\theta) = -\frac{1}{M} \sum_{k=1}^M \log P_\theta(\{s_i^{(k)}\})
$$

Substituting the Boltzmann distribution for $$ P_\theta $$, we get:

$$
P_\theta(\{s_i\}) = \frac{e^{-\mathcal{H}(\{s_i\})}}{\mathcal{Z}}
$$

$$
\log P_\theta(\{s_i\}) = -\mathcal{H}(\{s_i\}) - \log \mathcal{Z}
$$

Thus, the negative log-likelihood becomes:

$$
\mathcal{L}(\theta) = \frac{1}{M} \sum_{k=1}^M \mathcal{H}(\{s_i^{(k)}\}) + \log \mathcal{Z}
$$

Here:  
- The first term accounts for the energy of observed configurations.  
- The second term, $$ \log \mathcal{Z} $$, is the log of the partition function, which depends on $$ \theta $$ and involves summing over all possible spin configurations.

---

## Gradient of the Loss Function

To optimize $$ \mathcal{L}(\theta) $$, we calculate the gradients with respect to $$ h_i $$ and $$ J_{ij} $$. For simplicity, denote $$ \mathcal{L}(\theta) $$ as $$ \mathcal{L} $$.

Gradient w.r.t $$H_{i}$$,


$$
\frac{\partial \mathcal{L}}{\partial h_i} = \frac{\partial}{\partial h_i} \left( \frac{1}{M} \sum_{k=1}^M \mathcal{H}(\{s_i^{(k)}\}) + \log \mathcal{Z} \right)
$$

Using $$ \mathcal{H}(\{s_i\}) = -\sum_{j > i} J_{ij} s_i s_j - \sum_i h_i s_i $$, the derivative becomes:

$$
\frac{\partial \mathcal{H}}{\partial h_i} = -s_i
$$

Thus:

$$
\frac{\partial \mathcal{L}}{\partial h_i} = -\frac{1}{M} \sum_{k=1}^M s_i^{(k)} + \frac{\partial \log \mathcal{Z}}{\partial h_i}
$$

The partition function gradient is:


$$
\frac{\partial \log \mathcal{Z}}{\partial h_i} = \langle s_i \rangle_\theta
$$
Combining terms:

$$
\frac{\partial \mathcal{L}}{\partial h_i} = \langle s_i \rangle_\theta - \langle s_i \rangle^D
$$

Gradient w.r.t $$J_{ij}$$

$$
\frac{\partial \mathcal{L}}{\partial J_{ij}} = -\frac{1}{M} \sum_{k=1}^M s_i^{(k)} s_j^{(k)} + \langle s_i s_j \rangle_\theta
$$

---

## Parameter Update Rules

Using gradient descent to minimize $$ \mathcal{L}(\theta) $$, the updates for $$ h_i $$ and $$ J_{ij} $$ are:

$$
h_i^{n+1} = h_i^n + \eta \left( \langle s_i \rangle^D - \langle s_i \rangle_\theta \right)
$$

$$
J_{ij}^{n+1} = J_{ij}^n + \eta \left( \langle s_i s_j \rangle^D - \langle s_i s_j \rangle_\theta \right)
$$

where:  
- $$ \langle s_i \rangle^D $$: Empirical average of $$ s_i $$ from data.  
- $$ \langle s_i \rangle_\theta $$: Model average of $$ s_i $$ under the current parameters.  
- $$ \eta $$: Learning rate controlling the update step size.

---

The learning, in this case can be summarized as follows,


1. **Initialization**: Start with random parameters $$ h_i $$ and $$ J_{ij} $$.  
2. **Sampling**: Generate model samples $$ \{s_i\} $$ using techniques like Gibbs sampling or contrastive divergence to approximate $$ \langle s_i \rangle_\theta $$ and $$ \langle s_i s_j \rangle_\theta $$.  
3. **Compute Gradients**: Use the empirical data to compute $$ \langle s_i \rangle^D $$ and $$ \langle s_i s_j \rangle^D $$, and calculate the gradients.  
4. **Update Parameters**: Adjust $$ h_i $$ and $$ J_{ij} $$ using the update rules.  
5. **Iterate**: Repeat until, where $$ P_\theta(\{s_i\}) $$ closely matches the empirical $$ P_0(\{s_i\}) $$.

---

## Implementation in Python

Using the update rules mentioned above we can infer the target parameters. First we import neccessary libraries.

{% highlight python %}
#import necessary packages.
import scienceplots #better plots
import numpy as np
import matplotlib.pyplot as plt ; import matplotlib
import pandas as pd 
from tqdm import tqdm
matplotlib.rcParams['figure.dpi']=200
plt.style.use(['science','grid','no-latex'])
{% endhighlight %}


Define the one dimensional lattice and assign spins to it accordingly.


{% highlight python %}
#Create random spin 1D chains
def init_lattice(L):
  '''
  args : L = lattice size (L)
  returns : ndarray shape L
  '''
  lattice = 2 * np.random.randint(0, 2, L) - 1 ##arrange [-1,+1] spins accordingly
  return lattice
{% endhighlight %}

Now we need to create the target parameters that we are going to infer. To this end, we first 
create a function to create asymmetric and symmetric coupling matrix. 

{% highlight python %}
#Create target parameters that will be later inferred.
def create_target_params(N,assym=False):
  '''
  args : N = int, size of the interaction matrix , NxN and Nx1
         assym : flag = bool, set true for assymetric couplings.

  returns : ndarray with shapes NxN and Nx1
  
  '''  
  if assym==False:
    temp_matrix = np.random.normal(0,1/N,(N,N))
    J = 0.5*(temp_matrix + temp_matrix.T) ; np.fill_diagonal(J,0) # symmetric matrix with zero diagonal
    h = np.random.normal(0,1,size=N) 

  else:
    J = np.random.normal(0,1/N,(N,N)) ;  # Assymetric matrix
    np.fill_diagonal(J,0)
    h = np.random.normal(0,1,size=N)
  
  return J,h
{% endhighlight %}

Now we need to create a system which is thermalized. To do this, we thermalize the system using a Markov Chain - Monte Carlo (MCMC). A metropolis-Hastings step is performed in each markov step.

{% highlight python%}
#Calculate energy difference.

def calculate_energy_diff(temp_index,configuration,J,h):
  '''
  args : temp_index = int, temporary index for placeholder
         configuration = ndarray Nx1, spin configuration
         J = ndarray NxN, interaction matrix 
         h = ndarray Nx1, external field.

  return : E, float, energy difference between spin flip
  
  '''
    
  E = 2*configuration[temp_index] * (np.dot(J[temp_index,:],configuration) + h[temp_index])
  return E


#Metropolis dynamics.
 
def metropolis_dynamics(lattice,J,h):
  '''
  args : lattice = ndarray Nx1 , spin configuration 
         J = ndarray NxN, interaction matrix 
         h = ndarray Nx1, external field. 
         
  returns : configuration = ndarray Nx1, updated configuration
  ''' 
    
  configuration = lattice.copy() ; L = int(len(configuration))
  for _ in range(len(lattice)):
    random_index = np.random.randint(0,L)
    delta_E = calculate_energy_diff(random_index,configuration,J,h)

    prob = np.exp(-delta_E)
    if np.random.uniform(0,1) < prob:
      configuration[random_index] *= -1

  return configuration

{% endhighlight%}

Now we perform MCMC,

{%highlight python%}

#Perform markov chain monte carlo.
def markovchain_montecarlo(lattice,J,h,sweeps,burn_in):
  '''
  args : lattice = ndarray Nx1 , spin configuration 
         J = ndarray NxN, interaction matrix 
         h = ndarray Nx1, external field
         sweeps = int, number of sweeps
         burn_in = int, number of burn_in sweeps

  returns : mag = ndarray Nx1 , Mean Magnetisations
            C_ij = ndarray NxN, correlation matrix
            states = ndarray Nxsweeps, Final spins states.
  '''
    
    
  configuration = lattice.copy() ; L = len(lattice)
  C_ij = np.zeros((L,L))
  states = []

  for _ in range(burn_in):
    configuration = metropolis_dynamics(configuration,J,h)

  for sweep in range(sweeps):
    configuration = metropolis_dynamics(configuration,J,h)
    C_ij += np.outer(configuration,configuration)
    states.append(configuration)

  mag = np.mean(states,axis=0)
  C_ij /= sweeps

  return mag,C_ij,states

{% endhighlight %}


Note that we are saving both Magnetisations, and Spin-Spin correlations. The `states` will be used to find the empirical probability
distribution and in turn to find the negative log likelihood.


To check if the system thermalizes and to calculate the 'burn-in' time, we let the system runs with a range of values.

{% highlight python %}

J,h = create_target_params(5)
lattice = init_lattice(5)
sweeps = 10000

burn_in_steps = np.arange(20) # Make a list of burn in values
burn_ins = 2 ** burn_in_steps #

L = len(lattice)

M = np.zeros((len(burn_ins), L ))
C = np.zeros((len(burn_ins), L**2))

for i in range(len(burn_ins)):
  mag,C_ij,states = markovchain_montecarlo(lattice,J,h,burn_ins[i],0)
  M[i,:] = mag
  C[i,:] = C_ij.flatten()

#Take their differences so that it can be plotted.
Mags = abs(np.diff(M,axis=0)) 
Corrs = abs(np.diff(C,axis=0))

{% endhighlight %}


If we plot the results we see it clearly.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Boltzmann_plots/Thermalisations_boltzmann.png" title="Example result" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Thermalisation check.
</div>

Now we create neccessary functions to calculate some key parameters. First we need to observe how the learning 
process. To do this, we can calculate the negative log likelihood (NLL). To do this, we use the previously calculated 
spin configuration vectors.

{%highlight python%}

#Calculate Negative log likelihood
def calculate_NLL(configs):
    '''
    args : configs, list, list of final states obtained.
    returns : NLL, float, Negative log likelihood
    '''
    M = len(configs)
    unique_samples, sample_counts = np.unique(configs, axis=0, return_counts=True)
    Prob_distr = sample_counts / M
    NLL = -(1/M) * (np.sum(sample_counts*np.log(Prob_distr)))
    return NLL
{% endhighlight %}


Now we define the update steps for $$h$$ and $$J$$.

{% highlight python %}
# Update schemes
def h_update(h,mag_model,mag_train,eta):
  '''
  args : h = ndarray Nx1, external fields  
         mag_model = ndarray Nx1, Model magnetistaions
         mag_train = ndarray Nx1, training data
         eta = float, learning rate.

  returns : h_up = ndarray Nx1, updated parameter values 
  '''  
    
  h_up = h + eta * (mag_train - mag_model)
  return h_up

def J_update(J,corr_model,corr_train,eta):
  '''
  args : J = ndarray NxN, interaction matrix  
         corr_model = ndarray NxN, Model correlations
         corr_train = ndarray NxN, training data
         eta = float, learning rate.

  returns : J_up = ndarray NxN, updated parameter values 
  '''    
  J_up = J + eta * (corr_train - corr_model)
  np.fill_diagonal(J_up,0)
  return J_up
{% endhighlight %}

After the update schemes are defined, the final learning loop can be constructed.

{% highlight python %}
#Main learning block
def boltzmann_machine(L,iterations,sweeps,burn_in,mag_train,corr_train,eta):
  '''
  args : 
         L = int, size of the spin configuration chain
         iterations = int, number of iterations
         sweeps = int, number of markov sweeps
         burn_in = int, number of burn_in sweeps
         mag_train = ndarray Nx1, training data
         corr_train = ndarray NxN, training data
         eta = float, learning rate.

  returns : J = ndarray NxN, inferred interaction matrix
            h = ndarray Nx1, inferred field
            log_likelihoods = list, Negative log likelihoods
  '''  

    
  J,h = create_target_params(L)
  log_likelihoods = []
  for iter in tqdm(range(iterations)):
    mag_model,corr_model,states = markovchain_montecarlo(init_lattice(L),J,h,sweeps,burn_in)
    h = h_update(h,mag_model,mag_train,eta)
    J = J_update(J,corr_model,corr_train,eta)
    NLL = calculate_NLL(states)
    log_likelihoods.append(NLL)
  return J,h,log_likelihoods
{% endhighlight%}

Done! now all we have to do see if all of this works, is to generaete some sample values for $$h$$ and $$J$$, thermalize the system
and we can infer those generated values. Here I am plotting the learning process by varying $$\eta$$, the learning rate.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Boltzmann_plots/NLL_etas_boltzmann.png" title="Example result" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Negative log likelihood for different learning rates.
</div>

The inferred parameters look like this,

<div class="row">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Boltzmann_plots/J_etas_boltzmann.png" title="Example result 1" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Boltzmann_plots/h_etas_boltzmann.png" title="Example result 2" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    J and h inferred and actual values.
</div>


Note that $$J$$ was a 5x5 matrix, so it is flattened to look at the inference.  




Thumbnail image credits:

By <a href="//commons.wikimedia.org/w/index.php?title=User:HeMath&amp;action=edit&amp;redlink=1" class="new" title="User:HeMath (page does not exist)">HeMath</a> - <span class="int-own-work" lang="en">Own work</span>, <a href="https://creativecommons.org/licenses/by-sa/4.0" title="Creative Commons Attribution-Share Alike 4.0">CC BY-SA 4.0</a>, <a href="https://commons.wikimedia.org/w/index.php?curid=37327967">Link</a>