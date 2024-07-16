---
file_format: mystnb
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

Overview
========

Introduction
------------

### Who is this package for?

Cleo (Closed-Loop, Electrophysiology, and Optophysiology experiment simulation testbed) is a Python package developed to bridge theory and experiment for mesoscale neuroscience.
We envision two primary uses cases:

1.  For prototyping closed-loop control of neural activity *in silico*.
Animal experiments are costly to set up and debug, especially with the added complexity of real-time intervention---our aim is to enable researchers, given a decent spiking model of the system of interest, to assess whether the type of control they desire is feasible and/or what configuration(s) would be most conducive to their goals.
2.  The complexity of experimental interfaces means it's not always clear what a model would look like in a real experiment.
Cleo can help anyone interested in observing or manipulating a model while taking into account the constraints present in real experiments.
Because Cleo is built around the [Brian simulator](https://brian2.rtfd.io), we especially hope this is helpful for existing Brian users who for whatever reason would like a convenient way to inject recorders (e.g., electrodes or 2P imaging) or stimulators (e.g., optogenetics) into the core network simulation.

```{admonition} What is closed-loop control?
In short, determining the inputs to deliver to a system from its outputs.
In neuroscience terms, making the stimulation parameters a function of the data recorded in real time.
```

### Structure and design

Cleo wraps a spiking network simulator and allows for the injection of stimulators and/or recorders. The models used to emulate these devices are often non-trivial to implement or use in a flexible manner, so Cleo aims to make device injection and configuration as painless as possible, requiring minimal modification to the original network.

Cleo also orchestrates communication between the simulator and a user-configured {class}`~cleo.IOProcessor` object, modeling how experiment hardware takes samples, processes signals, and controls stimulation devices in real time.

For an explanation of why we choose to prioritize spiking network models and how we chose Brian as the underlying simulator, see {ref}`overview:design rationale`.


```{admonition} Why closed-loop control in neuroscience?
Fast, real-time, closed-loop control of neural activity enables intervention in processes that are too fast or unpredictable to control manually or with pre-defined stimulation, such as sensory information processing, motor planning, and oscillatory activity.
Closed-loop control in a *reactive* sense enables the experimenter to respond to discrete events of interest, such as the arrival of a traveling wave or sharp wave ripple, whereas *feedback* control deals with driving the system towards a desired point or along a desired state trajectory.
The latter has the effect of rejecting noise and disturbances, reducing variability across time and across trials, allowing the researcher to perform inference with less data and on a finer scale.
Additionally, closed-loop control can compensate for model mismatch, allowing it to reach more complex targets where open-loop control based on imperfect models is bound to fail.
```


Installation
------------

Make sure you have Python ≥ 3.7, then use pip: `pip install cleosim`.

```{caution}
The name on PyPI is `cleosim` since `cleo` [was already taken](https://cleo.readthedocs.io/en/latest/), but in code it is still used as `import cleo`.
The other Cleo appears to actually be a fairly well developed package, so I'm sorry if you need to use it along with this Cleo in the same environment.
In that case, [there are workarounds](https://stackoverflow.com/a/55817170/6461032).
```

Or, if you're a developer, [install poetry](https://python-poetry.org/docs/) and run `poetry install` from the repository root.

Usage
-----

### Brian network model

The starting point for using Cleo is a Brian spiking neural network model of the system of interest. For those new to Brian, the [docs](https://brian2.rtfd.io) are a great resource. If you have a model built with another simulator or modeling language, you may be able to [import it to Brian via NeuroML](https://brian2tools.readthedocs.io/en/stable/user/nmlimport.html).

Perhaps the biggest change you may have to make to an existing model to make it compatible with Cleo's optogenetics and electrode recording is to give the neurons of interest coordinates in space. See the tutorials or the cleo.coords module for more info.

You'll need your model in a Brian {class}`~brian2.core.network.Network` object before you move on. E.g.,:

```{code-cell} python
import brian2 as b2
ng = b2.NeuronGroup( # a simple population of 100 LIF neurons
    500,
    """dv/dt = (-v - 70*mV + (500*Mohm)*Iopto + 2*xi*sqrt(tau_m)*mvolt) / tau_m : volt
    Iopto : amp""",
    threshold='v>-50*mV',
    reset='v=-70*mV',
    namespace={'tau_m': 20*b2.ms},
)
ng.v = -70*b2.mV
net = b2.Network(ng)
```

Your neurons need x, y, and z coordinates for Cleo's spatially defined recording and stimulation models to work:

```{code-cell} python
import cleo
cleo.coords.assign_coords_rand_rect_prism(
    ng, xlim=(-0.2, 0.2), ylim=(-0.2, 0.2), zlim=(0.2, 1), unit=b2.mm
)
```


### CLSimulator

Once you have a network model, you can construct a {class}`~cleo.CLSimulator` object:

```{code-cell} ipython3
sim = cleo.CLSimulator(net)
```

The simulator object wraps the Brian network and coordinates device injection, processing input and output, and running the simulation.

### Recording

Recording devices take measurements of the Brian network.
To use a {class}`~cleo.Recorder`, you must inject it into the simulator via {meth}`cleo.CLSimulator.inject`.
The recorder will only record from the neuron groups specified on injection, allowing for such scenarios as singling out a cell type to record from.
Some extremely simple implementations (which do little more than wrap Brian monitors) are available in the {mod}`cleo.recorders` module.
See the {doc}`tutorials/electrodes` and {doc}`tutorials/all_optical` tutorials for more detail on how to record from a simulation more realistically, but here's a quick example of how to record multi-unit spiking activity with an electrode:

```{code-cell} ipython3
# configure and inject a 32-channel shank
coords = cleo.ephys.linear_shank_coords(
    array_length=1*b2.mm, channel_count=32, start_location=(0, 0, 0.2)*b2.mm
)
mua = cleo.ephys.MultiUnitSpiking(
    r_perfect_detection=20 * b2.um,
    r_half_detection=40 * b2.um,
)
probe = cleo.ephys.Probe(coords, signals=[mua])
sim.inject(probe, ng)
```

### Stimulation

{class}`~cleo.Stimulator` devices manipulate the Brian network, and are likewise {meth}`~cleo.CLSimulator.inject`ed into the simulator, into specified neuron groups.
Optogenetics (1P and 2P) is the main stimulation modality currently implemented by Cleo.
This requires injection of both a light source and an opsin---see the {doc}`tutorials/optogenetics` and {doc}`tutorials/all_optical` tutorials for more detail.

```{code-cell} ipython3
fiber = cleo.light.Light(
    coords=(0, 0, 0.5)*b2.mm,
    light_model=cleo.light.fiber473nm(),
    wavelength=473*b2.nmeter
)
chr2 = cleo.opto.chr2_4s()
sim.inject(fiber, ng).inject(chr2, ng)
```

```{note}
Markov opsin kinetics models require target neurons to have membrane potentials in realistic ranges and an `Iopto` term defined in amperes.
If you need to interface with a model without these features, you may want to use the simplified {class}`~cleo.opto.ProportionalCurrentOpsin`.
You can find more details, including a comparison between the two model types, in the {ref}`optogenetics tutorial <tutorials/optogenetics:Appendix: alternative opsin and neuron models>`.

```

```{note}
Recorders and stimulators need unique names which serve as keys to access/set values in the {class}`~cleo.IOProcessor`'s input/output dictionaries.
The name defaults to the class name, but you can specify it on construction.
```

### I/O Processor

Just as in a real experiment where the experiment hardware must be connected to signal processing equipment and/or computers for recording and control, the {class}`~cleo.CLSimulator` must be connected to an {class}`~cleo.IOProcessor`.
If you are only recording, you may want to use the {class}`~cleo.ioproc.RecordOnlyProcessor`.
Otherwise you will want to implement the {class}`~cleo.ioproc.LatencyIOProcessor`, which not only takes samples at the specified rate, but processes the data and delivers input to the network after a user-defined delay, emulating the latency inherent in real experiments.
This is done by creating a subclass and defining the {meth}`~cleo.ioproc.LatencyIOProcessor.process` function:

```{code-cell} ipython3
class MyProcessor(cleo.ioproc.LatencyIOProcessor):
    def process(self, state_dict, t_samp):
        # state_dict contains a {'recorder_name': value} dict of network.
        i_spikes, t_spikes, y_spikes = state_dict['Probe']['MultiUnitSpiking']
        # on-off control
        irr0 = 5 if len(i_spikes) < 10 else 0
        # output is a {'stimulator_name': value} dict and output time
        return {'Light': irr0 * b2.mwatt / b2.mm2}, t_samp + 3 * b2.ms  # (3 ms delay)

sim.set_io_processor(MyProcessor(sample_period=1 * b2.ms))
```

The {doc}`tutorials/on_off_ctrl`, {doc}`tutorials/PI_ctrl`, and {doc}`tutorials/lqr_ctrl_ldsctrlest` tutorials give examples of closed-loop control ranging from simple to complex.

### Visualization
{func}`cleo.viz.plot` allows you to easily visualize your experimental configuration:

```{code-cell} ipython3
:tags: [remove-cell]
cleo.utilities.style_plots_for_docs()
b2.prefs.codegen.target = 'numpy'
```

```{code-cell} ipython3
cleo.viz.plot(ng, colors=['#c500cc'], sim=sim, zlim=(200, 1000))
```

Cleo also features some {doc}`video visualization capabilities <tutorials/video_visualization>`.


### Running experiments

Use {meth}`cleo.CLSimulator.run` function with the desired duration.
This wrap's Brian's {meth}`brian2.core.network.Network.run` function:

```{code-cell} ipython3
:tags: [hide-output]
sim.run(50 * b2.ms)  # kwargs are passed to Brian's run function
```

Use {meth}`~cleo.CLSimulator.reset` to restore the default state (right after initialization/injection) for the network and all devices.
This could be useful for running a simulation multiple times under different conditions.

To facilitate access to data after the simulation, devices offer a {attr}`cleo.InterfaceDevice.save_history` option on construction, by default `True`.
If true, that object will store relevant variables as attributes.
For example:

```{code-cell} ipython3
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.scatter(mua.t / b2.ms, mua.i, marker='.', c='white', s=2)
ax1.set(ylabel='channel index', title='spikes')
ax2.step(fiber.t / b2.ms, fiber.values, c='#72b5f2')
ax2.set(xlabel='time (ms)', ylabel='irradiance (mW/mm²)', title='photostimulation')
```

Design rationale
----------------

### Why not prototype with more abstract models?

Cleo aims to be practical, and as such provides models at the level of abstraction corresponding to the variables the experimenter has available to manipulate.
This means models of spatially defined, spiking neural networks.

Of course, neuroscience is studied at many spatial and temporal scales.
While other projects may be better suited for larger segments of the brain and/or longer timescales (such as [HNN](https://elifesciences.org/articles/51214) or BMTK's [PopNet](https://alleninstitute.github.io/bmtk/popnet.html) or [FilterNet](https://alleninstitute.github.io/bmtk/filternet.html)), this project caters to finer-grained models because they can directly simulate the effects of alternate experimental configurations.
For example, how would the model change when swapping one opsin for another, using multiple opsins simultaneously, or with heterogeneous expression? How does recording or stimulating one cell type vs. another affect the experiment? Would using a more sophisticated control algorithm be worth the extra compute time, and thus later stimulus delivery, compared to a simpler controller?

Questions like these could be answered using an abstract dynamical system model of a neural circuit, but they would require the extra step of mapping the afore-mentioned details to a suitable abstraction---e.g., estimating a transfer function to model optogenetic stimulation for a given opsin and light configuration.
Thus, we haven't emphasized these sorts of models so far in our development of Cleo, though they should be possible to implement in Brian if you are interested.
For example, one could develop a Poisson linear dynamical system (PLDS), record spiking output, and configure stimulation to act directly on the system's latent state.

And just as experiment prototyping could be done on a more abstract level, it could also be done on an even more realistic level, which we did not deem necessary.
That brings us to the next point...

### Why Brian?

Brian is a relatively new spiking neural network simulator written in Python.
Here are some of its advantages:

-   Flexibility: allowing (and requiring!) the user to define models mathematically rather than selecting from a pre-defined library of cell types and features.
This enables us to define arbitrary models for recorders and stimulators and easily interface with the simulation
-   Ease of use: it's all just Python
-   Speed (at least when compiled to C++ or GPU---see [Brian2GENN](https://github.com/brian-team/brian2genn), [Brian2CUDA](https://github.com/brian-team/brian2cuda))

[NEST](https://www.nest-simulator.org/) is a popular alternative to Brian also strong in point neuron simulations.
However, it appears to be less flexible, and thus harder to extend.
[NEURON](https://www.neuron.yale.edu/neuron/) is another popular alternative to Brian.
Its main advantage is its first-class support of detailed, morphological, multi-compartment neurons.
In fact, strong alternatives to Brian for this project were BioNet ([docs](https://alleninstitute.github.io/bmtk/bionet.html), [paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0201630)) and NetPyNE ([docs](http://netpyne.org/index.html), [paper](https://elifesciences.org/articles/44494)), which already offer a high-level interface to NEURON with extracellular potential recording.
Optogenetics could be incorporated with [pre-existing .hoc code](https://github.com/ProjectPyRhO/PyRhO/blob/master/pyrho/NEURON/RhO4c.mod), though the light model would need to be implemented.
From brief examination of the [source code of BioNet](https://github.com/AllenInstitute/bmtk/blob/8c235eabbfa963a3fe163d6ba6e5ad67ca5ad7c3/bmtk/simulator/bionet/modules/sim_module.py#L44), it appears that closed-loop stimulation would not be too difficult to add.
It is unclear for NetPyNE.

In the end, we chose Brian since our priority was to model circuit/population-level dynamics over molecular/intra-neuron dynamics.
Also, Brian does have support for multi-compartment neurons, albeit less fully featured, if that is needed.

[PyNN](https://pynn.readthedocs.io/) would have been an ideal interface to build around, as it supports multiple simulator backends.
The difficulty is, implementing objects not natively supported by SNN simulators (e.g., opsins, calcium indicators, and light source) has required bespoke, idiosyncratic code applicable only to one simulator.
To do so in a native, efficient way, as we have attempted to do with Brian, would require significant work.
A collaborative effort extending a multi-simulator framework such as PyNN for this purpose may be of value if there is enough community interest in expanding the open-source SNN experiment simulation toolbox.

Future development
------------------

Here are some features which are missing but could be useful to add:

-   Electrode microstimulation
-   A more accurate LFP signal (only usable for morphological neurons) based on the volume conductor forward model as in [LFPy](https://lfpy.readthedocs.io/en/latest/index.html) or [Vertex](https://github.com/haeste/Vertex_2)
-   Voltage indicators
-   An expanded calcium indicator library---currently the parameter set for the full [NAOMi](https://bitbucket.org/adamshch/naomi_sim/src/master/) model is only available for GCaMP6f.
The [phenomenological S2F model](https://www.nature.com/articles/s41586-023-05828-9) should be easy to implement and fit to data, has parameters for several of the latest and greatest GECIs (jGCaMP7 and jGCaMP8 varieties), and should be cheaper to simulate as well.