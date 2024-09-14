---
layout: page
title: Neural synchrony.
date: 2023-04-24 21:01:00
description: Simulating synchrony in brain --> The hard way!
tags: Synchronisation, Brain
img: assets/img/Neural_sync/Neural_synchrony_project_thumb.png
importance: 2
category: Computational neuroscience
related_publications: false

---
 
Synchronization is a remarkable emergent phenomenon observed in many natural systems, where independent components begin to exhibit collective behavior. This phenomenon can be found in biological, physical, and even social systems. For instance, fireflies synchronize their flashing, neurons in the brain fire together in rhythmic patterns, and even coupled pendulums tend to swing in unison over time. These systems, often described by nonlinear dynamics, display synchronization when individual elements interact with each other, leading to a collective coherence despite differences in initial conditions or external influences. 

From the title, it should be clear to the reader that this article will detail an aspect of synchrony in the brain. Although it may be surprising to those new to the topic, synchrony is well-researched in the context of brain dynamics. In this article, I will outline how such a system can be studied through simulations. The full code (in an interactive Python notebook (.ipynb) file) is available in my GitHub repository: <a href="https://github.com/Cup-cake-lover/Neural-Synchrony.git">Neural-synchrony</a>.

## Synchronisation in dynamical systems

What is synchronisation? In the introduction, I briefly mentioned that it is an emergent phenomenon that appears in coupled dynamical systems. I won't be delving into a rigorous mathematical description of synchronisation in coupled systems. However, I will briefly explain it through some simulations I prepared using the open-source Python package <a href="https://www.manim.community/">`manim`</a>.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <iframe width="100%" height="315" src="https://www.youtube.com/embed/ZGvtnE1Wy6U" 
            frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
            allowfullscreen>
        </iframe>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        <iframe width="100%" height="315" src="https://www.youtube.com/embed/T58lGKREubo" 
            frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
            allowfullscreen>
        </iframe>
    </div>
</div>

<div class="caption">
    Common examples of synchronisation; Left shows synchrony of blinking patterns of fireflies. Right is the classic clicking metronomes.
</div>

### The Kuramoto model

Here, we consider a relatively simple model of synchrony: the Kuramoto model. The Kuramoto model describes a set of phase oscillators coupled through a sinusoidal function. The dynamics of the system of oscillators can then be described by the following differential equation.

$$
\dot \theta_i = \omega_i + \frac{k}{N} \sum_{j=1}^N \sin(\theta_j - \theta_i), ~~~i=1,\ldots, N
$$

Here $$\theta_i$$ represents phase of the $$i^{th}$$ oscillator. The coupling strength between these oscillators are described by the coupling constant $$k$$. The value of $$k$$ is crucial in the synchronisation behaviour of the system. So this describes our model, how do we charecterize synchrony? One way is to define an order parameter $$R$$. It is defined as follows,

$$
R\equiv   re^{i\psi} = \frac{1}{N}\sum_{j=1}^N e^{i \theta_j}.
$$


Order parameters are usually a statistical physicist's 'favorite' quantity to compute when the collective behavior of a system undergoes rapid change. In the animations shown below, both the $$K$$ value and the corresponding $$R$$ value are displayed. Notice that for very low coupling, the order parameter never increases. However, when we set $$K>0$$, we observe that the order parameter rises drastically, indicating that the system is achieving synchrony.
 
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid path="assets/video/neural_sync/KuramotoOscillatorsk_0.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid path="assets/video/neural_sync/KuramotoOscillatorss0.7.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=true %}
    </div>
</div>
<div class="caption">
    Manim animations of a system of Kuramoto oscillators. The animation on the left represents the system with coupling value set to zero, on the right, the coupling value is non zero, resulting in the synchrony of oscillators.
</div>


## Synchrony in the brain

Our brain is composed of millions of neurons, synapses, and associated cells. Together, they form an incredible computing unit that performs complex computations of the real world. With its help, we receive, process, and communicate information. What fascinates me even more is the sheer complexity of the brain. Not only does it maintain high-level processes such as thoughts and memory, but it is also an integral part of low-level, involuntary operations that happen every second in our body. Researchers have been studying the brain as a complex collective dynamical system to understand how individual computing units composed of neurons and synapses work together.

Now, after all the foreshadowing from the previous section, we finally reach the point where I will discuss neural synchrony. The brain is composed of inhibitory and excitatory neurons. As their names suggest, inhibitory neurons inhibit the actions of excitatory neurons. We consider networks of such neurons with varying percentage-wise compositions. The network connections are generally defined probabilistically, meaning we set a probability value for connections between excitatory-excitatory ($$p_{ee}$$), inhibitory-inhibitory ($$p_{ii}$$), inhibitory-excitatory ($$p_{ie}$$) and excitatory-inhibitory ($$p_{ei}$$) connections. 

Computationally modeling such systems has become relatively simpler in recent years. Dedicated neural dynamics packages like <a href="https://www.nest-simulator.org/">`NEST`</a> or <a href="https://brian2.readthedocs.io/en/stable/">`Brian2`</a> exist for advanced large-scale simulations. It is possible to generate the results I am going to show in this article using such packages. However, I want to take a more 'coded from scratch' approach. While this is a challenging task, I believe it provides much more conceptual insight.

To begin, we need a neuron model to describe how a single neuron operates in our system. I will follow the approach by <a href="https://doi.org/10.3389/fncom.2021.663408">Protachevicz et al. 2021</a> from their paper on the 'Emergence of Neuronal Synchronisation in Coupled Areas'. In this paper, they study the coupling dynamics of two different brain regions composed of two distinct networks. Here, we consider a piece of the cortex network, which is largely composed of pyramidal excitatory neurons and inhibitory interneurons. The excitatory neurons exhibit an adaptive spiking mechanism, while inhibitory neurons do not. This behavior can be modeled by the Adaptive Exponential Leaky Integrate-and-Fire (AdEx-LIF) neuron model. Mathematically, the model can be written as follows:

$$
C_m \frac{dV_j}{dt} = g_L (V_j - E_L) + g_L \Delta T \exp\left(\frac{V_j - V_T}{\Delta T}\right) - w_j + I + I_{j}
$$

$$ 
\tau_w \frac{dw_j}{dt} = a_j (V_j - E_L) - w_j 
$$

$$
\tau_s \frac{dg_j}{dt} = -g_j
$$

And the spiking dynamics is achieved by the following update dynamics,

$$
V_j \rightarrow V_r
$$

$$
w_j \rightarrow w_j + b_j \\
$$

$$
g_j \rightarrow g_j + g_s
$$

Here, the membrane potential of a neuron is denoted as $$V_j$$, where $$C_m$$ is the membrane capacitance, $$g_L$$ represents the leak conductance, and $$E_L$$ is the leak reversal potential. The exponential term involves the slope factor $$\Delta T$$ and the spike threshold potential $$V_T$$. The adaptation current for each neuron is $$w_j$$, and the parameters governing the adaptation mechanism are the subthreshold adaptation $$a_j$$, the triggered adaptation $$b_j$$, and the adaptation time constant $$\tau_w$$. The injected current into the neuron is $$I$$, and the synaptic input is represented by the synaptic conductance $$g_{jk}$$ and the synaptic time constant $$\tau_s$$. The chemical input current is $$I_{\text{chem}}$$, and the synaptic reversal potential is $$V_{\text{REV}}$$. Synaptic conductances for connections between excitatory and inhibitory neurons are denoted as $$g_{ee}$$, $$g_{ei}$$, $$g_{ie}$$, and $$g_{ii}$$. 

Now that we have a model, we can set this up in code. To do this, different paramter values needs to adapted. I am going to take the values mentioned in the paper as a stardarnd a write a dictionary in Python as following. As we are coding everythign from 'scratch', we are going to only use `numpy` for mathematical operations and `matplotlib` for plotting.

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
{% endhighlight %}


{% highlight python %}
def initialize_parameters():
    params = {
        "Cm": 200,        # Capacitance membrane (pF)
        "gL": 12,         # Leak conductance (nS)
        "EL": -70,        # Leak reversal potential (mV)
        "Delta_T": 2,     # Slope factor (mV)
        "d_exc": 1.5,     # Excitatory conductance time delay (ms)
        "d_inh": 0.8,     # Inhibitory conductance time delay (ms)
        "VT": -50,        # Spike threshold (mV)
        "I": 270,         # Injection current (pA)
        "aj_exc": np.random.uniform(1.9, 2.1) * 1,  # Subthreshold adaptation for excitatory neurons (nS)
        "bj_exc": 70,     # Spike-triggered adaptation for excitatory neurons (pA)
        "aj_inh": 0,      # Subthreshold adaptation for inhibitory neurons (nS)
        "bj_inh": 0,      # Spike-triggered adaptation for inhibitory neurons (pA)
        "tau_w": 300,     # Adaptation time constant (ms)
        "tau_s": 2.728,   # Synaptic time constant (ms)
        "Vr": -58,        # Reset potential (mV)
        "Vthres": -50,    # Threshold potential (mV)
        "Vexc_REV": 0,    # Excitatory reversal potential (mV)
        "Vinh_REV": -80,  # Inhibitory reversal potential (mV)
        "gee": 0.5,        # Synaptic conductance for excitatory to excitatory (nS)
        "gei": 2,       # Synaptic conductance for excitatory to inhibitory (nS)
        "gie": 1.5,        # Synaptic conductance for inhibitory to excitatory (nS)
        "gii": 2,        # Synaptic conductance for inhibitory to inhibitory (nS)
        "dt": 0.1,        # Time step (ms)
        "T": 1000,        # Total simulation time (ms)
        "N": 100,        # Number of neurons
        "Pexc": 0.8,      # Proportion of excitatory neurons
        "Pinh": 0.2,      # Proportion of inhibitory neurons
        "connection_prob": {
            "pee": 0.05,  # Excitatory to excitatory within area
            "pii": 0.2,   # Inhibitory to inhibitory within area
            "pei": 0.05,  # Excitatory to inhibitory within area
            "pie": 0.05,  # Inhibitory to excitatory within area
        }
    }
    return params
{% endhighlight %}


After the parameters are set we need to define a network in which the nodes are going to place the neurons we created. First we create a population of inhibitory and excitatory neurons:

{% highlight python %}
def initialize_neurons(params):
    N, P_exc = params['N'], params['Pexc']
    N_exc = int(N * P_exc)  # Number of excitatory neurons
    N_inh = N - N_exc       # Number of inhibitory neurons
    
    # Create an array with the first N_exc elements as 1 (excitatory) and the rest as -1 (inhibitory)
    neuron_types = np.array([1] * N_exc + [-1] * N_inh)
    
    return neuron_types
{% endhighlight %}

Now we describe dynamics of the neurons, to do this we first create functions that represents the differential equations of the adEx-LIF model as follows,

{% highlight python %}
def dVdt(V, w, I_chem):
    return (params['gL'] * (params['EL'] - V) + params['gL'] * params['Delta_T'] * np.exp((V - params['Vthres']) / params['Delta_T']) - w + I_chem + params['I']) / params['Cm']

def dwdt(V, w, neuron_types):
    a = np.where(neuron_types == 1, params['aj_exc'], 0)
    return (a * (V - params['EL']) - w) / params['tau_w']

def dg_syn_dt(g_syn):
    return -g_syn / params['tau_s']
{% endhighlight %}

To solve them iteratively, we use a numerical integration scheme, here we use a 4th order Runge Kutta method. Which is coded as follows,

{% highlight python %}
def rk4_step(V, w, g_syn_exc, g_syn_inh, dt, I_chem, neuron_types):
    k1_V = dVdt(V, w, I_chem)
    k1_w = dwdt(V, w, neuron_types)
    k1_g_syn_exc = dg_syn_dt(g_syn_exc)
    k1_g_syn_inh = dg_syn_dt(g_syn_inh)
    
    k2_V = dVdt(V + 0.5 * dt * k1_V, w + 0.5 * dt * k1_w, I_chem)
    k2_w = dwdt(V + 0.5 * dt * k1_V, w + 0.5 * dt * k1_w, neuron_types)
    k2_g_syn_exc = dg_syn_dt(g_syn_exc + 0.5 * dt * k1_g_syn_exc)
    k2_g_syn_inh = dg_syn_dt(g_syn_inh + 0.5 * dt * k1_g_syn_inh)
    
    k3_V = dVdt(V + 0.5 * dt * k2_V, w + 0.5 * dt * k2_w, I_chem)
    k3_w = dwdt(V + 0.5 * dt * k2_V, w + 0.5 * dt * k2_w, neuron_types)
    k3_g_syn_exc = dg_syn_dt(g_syn_exc + 0.5 * dt * k2_g_syn_exc)
    k3_g_syn_inh = dg_syn_dt(g_syn_inh + 0.5 * dt * k2_g_syn_inh)
    
    k4_V = dVdt(V + dt * k3_V, w + dt * k3_w, I_chem)
    k4_w = dwdt(V + dt * k3_V, w + dt * k3_w, neuron_types)
    k4_g_syn_exc = dg_syn_dt(g_syn_exc + dt * k3_g_syn_exc)
    k4_g_syn_inh = dg_syn_dt(g_syn_inh + dt * k3_g_syn_inh)
    
    V_new = V + (dt / 6.0) * (k1_V + 2.0 * k2_V + 2.0 * k3_V + k4_V)
    w_new = w + (dt / 6.0) * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w)
    g_syn_exc_new = g_syn_exc + (dt / 6.0) * (k1_g_syn_exc + 2.0 * k2_g_syn_exc + 2.0 * k3_g_syn_exc + k4_g_syn_exc)
    g_syn_inh_new = g_syn_inh + (dt / 6.0) * (k1_g_syn_inh + 2.0 * k2_g_syn_inh + 2.0 * k3_g_syn_inh + k4_g_syn_inh)
    
    return V_new, w_new, g_syn_exc_new, g_syn_inh_new
{% endhighlight %}


The model is set now, we move onto creating a network with specific connection probabilities.


{% highlight python %}
def make_network(params):
    neurons = initialize_neurons(params)
    N = params['N']
    connectivity_within = {
        "pee": np.random.rand(N, N) < params["connection_prob"]["pee"],
        "pii": np.random.rand(N, N) < params["connection_prob"]["pii"],
        "pei": np.random.rand(N, N) < params["connection_prob"]["pei"],
        "pie": np.random.rand(N, N) < params["connection_prob"]["pie"]
    }
  
    V = np.zeros(N)
    w = np.zeros(N)
    g_exc = np.zeros(N)
    g_inh = np.zeros(N)
    
    for i in range(N):
        V[i] = np.random.uniform(params['EL'],params['VT'])
        
    for j in range(N):
        w[j] = np.random.uniform(0,3) if neurons[j]==1 else 0
                
    return connectivity_within,V,w,g_exc,g_inh
{% endhighlight %}

Finally to tie everything up, we create a `simulate` function. Note that the dynamic differs for both inhibitory and excitatory neurons. This is taken care of by simply considering the label of the neuron we defined earlier. We set the total simulation time to be 3000 ms for a total of 100 neurons. The integration time step is set to be 0.02 ms.

{% highlight python %}
def simulate(params):
    connectivity_within,V,w,g_exc,g_inh = make_network(params)
    dt = 0.02
    T = 3000
    N = int(params['N'])
    neurons = initialize_neurons(params)
    simulation_time = int(T/dt)
    V_trace = np.zeros((simulation_time, N))
    spikes = np.zeros((simulation_time, N))
    for t in range(simulation_time):
        spike_time = 0
        for j in range(N):
            #Spike when crosses threshold
            if V[j] >= params['Vthres']:
                V[j] = params['Vr']
                w[j] += params['bj_exc'] if neurons[j]==1 else params['bj_inh']

                if neurons[j] == 1:
                    g_exc += params['gee']*connectivity_within['pee'][j,:]*connectivity_within['pei'][j,:]
                if neurons[j] == -1:
                    g_inh += params['gii']*connectivity_within['pii'][j,:]*connectivity_within['pie'][j,:]
                
                spike_time += t*0.02
                spikes[t,j] += 1
            Isyn_exc = g_exc[j]*(params['Vexc_REV'] - V[j])*(spike_time - params['d_exc'])
            Isyn_inh = g_inh[j]*(params['Vinh_REV'] - V[j])*(spike_time - params['d_inh'])
            
            I_chem = Isyn_exc + Isyn_inh
            
        V, w, g_exc, g_inh = rk4_step(V, w, g_exc, g_inh, dt, I_chem,neurons)
        V_trace[t,:] += V
    return V_trace,w,spikes
{% endhighlight %}

We plot the results uing matplotlib.



{% highlight python %}
dt = 0.02  # Time step in ms
T = 3000   # Total time in ms
spike_times = np.where(spikes == 1)[0]
neuron_indices = np.where(spikes == 1)[1]
spike_times_real = spike_times * dt
inh_neurons = neuron_indices >= 80  # Boolean mask for inhibitory neurons
plt.figure(figsize=(18,5))
plt.scatter(spike_times_real[inh_neurons], neuron_indices[inh_neurons], color='r', marker='.', label='Inhibitory Neurons')
plt.scatter(spike_times_real[~inh_neurons], neuron_indices[~inh_neurons], color='b', marker='.', label='Excitatory Neurons')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron Index')
plt.xticks(np.arange(0, T + 1, 200))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()  # Adjust layout to make room for the legend
plt.savefig('neuralsync.png',dpi=400)
{% endhighlight %}


For certain values of conductances $$g_{ee},g_{ii},g_{ie},g_{ei}$$. The spike patterns synchronize, as shown below.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Neural_sync/neuralsync.png" title="Synchronisation" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Neuronal synchrony; The blue represents the inhibitory which shows regular spiking, and exitatory neurons a synchronized spiking behaviour
</div>

So there we have it! A naive but direct look at synchronous spiking behaviour in the brain. Ofcourse, the code I have documented here has a lot issues, which includes stability problems and extremely long computation time for larger number of neurons. Hence, this is not at all an efficient way of coding it up. A cleaner implementation of this can be done by the aforementioned `NEST` of `Brian2` package.

Project tile image credits:
<a href="https://commons.wikimedia.org/wiki/File:Journal.pone.0082873.g001.png">Bumhee Park, Dae-Shik Kim, Hae-Jeong Park</a>, <a href="https://creativecommons.org/licenses/by-sa/4.0">CC BY-SA 4.0</a>, via Wikimedia Commons