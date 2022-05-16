###### CLEOSim: Closed Loop, Electrophysiology, and Optogenetics Simulator
[![Test and lint](https://github.com/kjohnsen/cleosim/actions/workflows/test.yml/badge.svg)](https://github.com/kjohnsen/cleosim/actions/workflows/test.yml)
[![Documentation Status](https://readthedocs.org/projects/cleosim/badge/?version=latest)](https://cleosim.readthedocs.io/en/latest/?badge=latest)


<h1>
<p align="center">
  <img 
      style="display: block; 
             width: 50%;"
      src="https://user-images.githubusercontent.com/19983357/167221164-33ca27e5-e2cb-4dd6-9cb7-2159e4a84b82.png" 
      alt="CLEOSim: Closed Loop, Electrophysiology, and Optogenetics Simulator">
  </img>
</p>
</h1>


Hello there! CLEOSim has the goal of bridging theory and experiment for mesoscale neuroscience, facilitating electrode recording, optogenetic stimulation, and closed-loop experiments (e.g., real-time input and output processing) with the [Brian 2](https://brian2.readthedocs.io/en/stable/) spiking neural network simulator. We hope users will find these components useful for prototyping experiments, innovating methods, and testing observations about a hypotheses *in silico*, incorporating into spiking neural network models laboratory techniques ranging from passive observation to complex model-based feedback control. CLEOSim also serves as an extensible, modular base for developing additional recording and stimulation modules for Brian simulations.

This package was developed by [Kyle Johnsen](https://kjohnsen.org) and Nathan Cruzado under the direction of [Chris Rozell](https://siplab.gatech.edu) at Georgia Institute of Technology.

<p align="center">
  <img 
      style="display: block; 
             width: 90%;"
      src="https://user-images.githubusercontent.com/19983357/167451424-5d04d3df-d8d0-42ae-9cc9-8b9a74da5eb2.png" 
      alt="logo">
  </img>
</p>

## <img align="top" src="https://user-images.githubusercontent.com/19983357/167456512-fb10619b-255e-4a53-8ed9-79ae954d3ff4.png" alt="CL icon" > Closed Loop processing
CLEOSim allows for flexible I/O processing in real time, enabling the simulation of closed-loop experiments such as event-triggered or feedback control. The user can also add latency to closed-loop stimulation to study the effects of computation delays.


## <img align="top" src="https://user-images.githubusercontent.com/19983357/167461111-b0a3746c-03fa-47b7-a9a9-7b651157044f.png" alt="CL icon" > Electrode recording
CLEOSim provides functions for configuring electrode arrays and placing them in arbitrary locations in the simulation. The user can then specify parameters for probabilistic spike detection or a spike-based LFP approximation developed by [Teleńczuk et al., 2020](https://www.sciencedirect.com/science/article/pii/S0165027020302946).

## <img align="top" src="https://user-images.githubusercontent.com/19983357/167461525-1f84e8ae-498b-4b52-9909-dade375f2006.png" alt="CL icon" > Optogenetic stimulation
By providing an optic fiber-light propagation model, CLEOSim enables users to flexibly add photostimulation to their model. Both a four-state Markov state model of opsin dynamics is available, as well as a minimal proportional current option for compatibility with simple neuron models. Parameters are provided for the common blue light/ChR2 setup.

## Getting started
Just use pip to install:
```
pip install cleosim
```

Then head to the [overview section of the documentation](https://cleosim.readthedocs.io/en/latest/overview.html) for a more detailed discussion of motivation, structure, and basic usage.

## Related resources
Those using CLEOSim to simulate closed-loop control experiments may be interested in software developed for the execution of real-time, *in-vivo* experiments. Developed by members of [Chris Rozell](https://siplab.gatech.edu)'s and [Garrett Stanley](https://stanley.gatech.edu/)'s labs at Georgia Tech, the [CLOCTools repository](https://cloctools.github.io) can serve these users in two ways:

1. By providing utilities and interfaces with experimental platforms for moving from simulation to reality.
2. By providing performant control and estimation algorithms for feedback control. Although CLEOSim enables closed-loop manipulation of network simulations, it does not include any advanced control algorithms itself. The `ldsCtrlEst` library implements adaptive linear dynamical system-based control while the `hmm` library can generate and decode systems with discrete latent states and observations.

<p align="center">
  <img 
      style="display: block; 
             width: 100%;"
      src="https://user-images.githubusercontent.com/19983357/167465825-363ad169-bc2e-412f-a8ab-12f960769e9b.png" 
      alt="CLOCTools and CLEOSim">
  </img>
</p>

### Publications
[**CLOC Tools: A Library of Tools for Closed-Loop Neuroscience**](https://github.com/cloctools/tools-for-neuro-control-manuscript)<br>
A.A. Willats, M.F. Bolus, K.A. Johnsen, G.B. Stanley, and C.J. Rozell. *In prep*, 2022.

[**State-Aware Control of Switching Neural Dynamics**](https://github.com/awillats/state-aware-control)<br>
A.A. Willats, M.F. Bolus, C.J. Whitmire, G.B. Stanley, and C.J. Rozell. *In prep*, 2022.

[**Closed-Loop Identifiability in Neural Circuits**](https://github.com/awillats/clinc)<br>
A. Willats, M. O'Shaughnessy, and C. Rozell. *In prep*, 2022.

[**State-space optimal feedback control of optogenetically driven neural activity**](https://www.biorxiv.org/content/10.1101/2020.06.25.171785v2)<br>
M.F. Bolus, A.A. Willats, C.J. Rozell and G.B. Stanley. *Journal of Neural Engineering*, 18(3), pp. 036006, March 2021.

[**Design strategies for dynamic closed-loop optogenetic neurocontrol in vivo**](https://iopscience.iop.org/article/10.1088/1741-2552/aaa506)<br>
M.F. Bolus, A.A. Willats, C.J. Whitmire, C.J. Rozell and G.B. Stanley. *Journal of Neural Engineering*, 15(2), pp. 026011, January 2018.
