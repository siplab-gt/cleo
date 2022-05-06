# CLEOsim: Closed Loop, Electrophysiology, and Optogenetics Simulator

[![Test and lint](https://github.com/kjohnsen/cleosim/actions/workflows/test.yml/badge.svg)](https://github.com/kjohnsen/cleosim/actions/workflows/test.yml)
[![Documentation Status](https://readthedocs.org/projects/cleosim/badge/?version=latest)](https://cleosim.readthedocs.io/en/latest/?badge=latest)

<p align="center">
  <img 
      style="display: block; 
             width: 50%;"
      src="https://user-images.githubusercontent.com/19983357/167221164-33ca27e5-e2cb-4dd6-9cb7-2159e4a84b82.png" 
      alt="logo">
  </img>
</p>

Hello there! CLEOsim has the goal of bridging theory and experiment for mesoscale neuroscience, facilitating electrode recording, optogenetic stimulation, and closed-loop experiments (e.g., real-time input and output processing) with the [Brian 2](https://brian2.readthedocs.io/en/stable/) spiking neural network simulator. We hope users will find these components useful for prototyping experiments, exploring methods, and testing observations about a hypothesis in silico, incorporating into spiking neural network models laboratory techniques ranging from passive observation to complex model-based feedback control. CLEOsim also serves as an extensible, modular base for developing additional recording and stimulation modules for Brian simulations.

This package was developed by [Kyle Johnsen](https://kjohnsen.org) and Nathan Cruzado under the direction of [Chris Rozell](https://siplab.gatech.edu) at Georgia Institute of Technology.

## Getting started
Just use pip to install:
```
pip install cleosim
```

For a more detailed discussion of motivation, structure, and basic usage, head to the [overview section](https://cleosim.readthedocs.io/en/latest/overview.html) of the documentation.

## Related resources
Those using CLEOsim to simulate closed-loop control experiments may be interested in software developed for the execution of real-time, *in-vivo* experiments. Developed by members of [Chris Rozell](https://siplab.gatech.edu)'s and [Garrett Stanley](https://stanley.gatech.edu/)'s labs at Georgia Tech, the [CLOCTools repository](https://github.com/stanley-rozell/cloctools/blob/main/README.md) can serve these users in two ways:

1. By providing utilities and interfaces with experimental platforms for moving from simulation to reality.
2. By providing performant control and estimation algorithms for feedback control. Although CLEOsim enables closed-loop manipulation of network simulations, it does not include any advanced control algorithms itself. The `ldsCtrlEst` library implements adaptive linear dynamical system-based control while the `hmm` library can generate and decode systems with discrete latent states and observations.

<p align="center">
  <img 
      style="display: block; 
             width: 90%;"
      src="https://user-images.githubusercontent.com/19983357/167221389-d2eb8dbe-5f36-46c3-b09e-d1c3b0e7d894.png" 
      alt="CLOCTools and CLEOsim">
  </img>
</p>
