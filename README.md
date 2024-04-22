<h1 style="font-size: 1em;">Cleo: the Closed-Loop, Electrophysiology, and Optophysiology experiment simulation testbed
</h1>

[![Tests](https://github.com/kjohnsen/cleosim/actions/workflows/test.yml/badge.svg)](https://github.com/kjohnsen/cleosim/actions/workflows/test.yml)
[![Documentation Status](https://readthedocs.org/projects/cleosim/badge/?version=latest)](https://cleosim.readthedocs.io/latest/?badge=latest)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![DOI](https://zenodo.org/badge/262130097.svg)](https://zenodo.org/doi/10.5281/zenodo.7036270)


<h1>
<p align="center">
  <img 
      style="display: block; 
             width: 50%;"
      src="https://user-images.githubusercontent.com/19983357/187561700-100b853a-d226-4039-a580-1d798b00f9e4.png" 
      alt="Cleo: the Closed-Loop, Electrophysiology, and Optophysiology experiment simulation testbed">
  </img>
</p>
</h1>


Hello there!
Cleo has the goal of bridging theory and experiment for mesoscale neuroscience, facilitating electrode recording, optogenetic stimulation, and closed-loop experiments (e.g., real-time input and output processing) with the [Brian 2](https://brian2.readthedocs.io/en/stable/) spiking neural network simulator.
We hope users will find these components useful for prototyping experiments, innovating methods, and testing observations about a hypotheses *in silico*, incorporating into spiking neural network models laboratory techniques ranging from passive observation to complex model-based feedback control.
Cleo also serves as an extensible, modular base for developing additional recording and stimulation modules for Brian simulations.

This package was developed by [Kyle Johnsen](https://kjohnsen.org) and Nathan Cruzado under the direction of [Chris Rozell](https://siplab.gatech.edu) at Georgia Institute of Technology.
See the preprint [here](https://www.biorxiv.org/content/10.1101/2023.01.27.525963).

<p align="center">
  <img 
      style="display: block; 
             width: 90%;"
      src="https://raw.githubusercontent.com/siplab-gt/cleo/master/docs/_static/cleo-overview-table.png" 
      alt="Overview table of Cleo's closed-loop control, ephys, and ophys features">
  </img>
</p>

## <img align="bottom" src="https://user-images.githubusercontent.com/19983357/167456512-fb10619b-255e-4a53-8ed9-79ae954d3ff4.png" alt="🖥️" > Closed Loop processing
Cleo allows for flexible I/O processing in real time, enabling the simulation of closed-loop experiments such as event-triggered or feedback control.
The user can also add latency to the stimulation to study the effects of computation delays.


## <img align="bottom" src="https://user-images.githubusercontent.com/19983357/167461111-b0a3746c-03fa-47b7-a9a9-7b651157044f.png" alt="🔌" > Electrode recording
Cleo provides functions for configuring electrode arrays and placing them in arbitrary locations in the simulation.
The user can then specify parameters for probabilistic spike detection or a spike-based LFP approximation developed by [Teleńczuk et al., 2020](https://www.sciencedirect.com/science/article/pii/S0165027020302946).

## <img align="bottom" src="https://user-images.githubusercontent.com/19983357/187728089-62fae854-1d69-4e8f-a597-a25934ca3eaa.png" alt="⚡" > 1P/2P optogenetic stimulation
By modeling light propagation and opsins, Cleo enables users to flexibly add photostimulation to their model.
Both a four-state Markov state model of opsin kinetics is available, as well as a minimal proportional current option for compatibility with simple neuron models.
Cleo also accounts for opsin action spectra to model the effects of multi-light/wavelength/opsin crosstalk and heterogeneous expression.
Parameters are for multiple opsins, and blue optic fiber (1P) and infrared spot (for 2P) illumination.

## <img src="https://github.com/siplab-gt/cleo/assets/19983357/08b473bf-7e19-4dfb-9a21-1f9772f7ed50" alt="🔬" > 2P imaging

Users can also inject a microscope into their model, selecting neurons on the specified plane of imaging or elsewhere, with signal and noise strength determined by indicator expression levels and position with respect to the focal plane.
The calcium indicator model of [Song et al., 2021](https://www.sciencedirect.com/science/article/pii/S0165027021001084) is implemented, with parameters included for GCaMP6 variants.

## 🚀 Getting started
Just use pip to install&mdash;the name on PyPI is `cleosim`:
```bash
pip install cleosim
```

Then head to the [overview section of the documentation](https://cleosim.readthedocs.io/latest/overview.html) for a more detailed discussion of motivation, structure, and basic usage.

## 📚 Related resources
Those using Cleo to simulate closed-loop control experiments may be interested in software developed for the execution of real-time, *in-vivo* experiments.
Developed by members of [Chris Rozell](https://siplab.gatech.edu)'s and [Garrett Stanley](https://stanley.gatech.edu/)'s labs at Georgia Tech, the [CLOCTools repository](https://cloctools.github.io) can serve these users in two ways:

1. By providing utilities and interfaces with experimental platforms for moving from simulation to reality.
2. By providing performant control and estimation algorithms for feedback control.
Although Cleo enables closed-loop manipulation of network simulations, it does not include any advanced control algorithms itself.
The `ldsCtrlEst` library implements adaptive linear dynamical system-based control while the `hmm` library can generate and decode systems with discrete latent states and observations.

<p align="center">
  <img 
      style="display: block; 
             width: 100%;"
      src="https://raw.githubusercontent.com/siplab-gt/cleo/master/docs/_static/cloctools_and_cleo.png" 
      alt="CLOCTools and Cleo">
  </img>
</p>

### 📃 Publications
[**Cleo: A testbed for bridging model and experiment by simulating closed-loop stimulation, electrode recording, and optophysiology**](https://www.biorxiv.org/content/10.1101/2023.01.27.525963)<br>
K.A. Johnsen, N.A. Cruzado, Z.C. Menard, A.A. Willats, A.S. Charles, and C.J. Rozell. *bioRxiv*, 2023.

[**CLOC Tools: A Library of Tools for Closed-Loop Neuroscience**](https://github.com/cloctools/tools-for-neuro-control-manuscript)<br>
A.A. Willats, M.F. Bolus, K.A. Johnsen, G.B. Stanley, and C.J. Rozell. *In prep*, 2023.

[**State-Aware Control of Switching Neural Dynamics**](https://github.com/awillats/state-aware-control)<br>
A.A. Willats, M.F. Bolus, C.J. Whitmire, G.B. Stanley, and C.J. Rozell. *In prep*, 2023.

[**Closed-Loop Identifiability in Neural Circuits**](https://github.com/awillats/clinc)<br>
A. Willats, M. O'Shaughnessy, and C. Rozell. *In prep*, 2023.

[**State-space optimal feedback control of optogenetically driven neural activity**](https://www.biorxiv.org/content/10.1101/2020.06.25.171785v2)<br>
M.F. Bolus, A.A. Willats, C.J. Rozell and G.B. Stanley. *Journal of Neural Engineering*, 18(3), pp. 036006, March 2021.

[**Design strategies for dynamic closed-loop optogenetic neurocontrol in vivo**](https://iopscience.iop.org/article/10.1088/1741-2552/aaa506)<br>
M.F. Bolus, A.A. Willats, C.J. Whitmire, C.J. Rozell and G.B. Stanley. *Journal of Neural Engineering*, 15(2), pp. 026011, January 2018.
