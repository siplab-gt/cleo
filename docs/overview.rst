Overview
========

Introduction
------------
Who is this package for?
^^^^^^^^^^^^^^^^^^^^^^^^
Cleo (Closed Loop, Electrophysiology, and Optogenetics Simulator) is a Python package developed to bridge theory and experiment for mesoscale neuroscience. We envision two primary uses cases:

1. For prototyping closed-loop control of neural activity *in silico*. Animal experiments are costly to set up and debug, especially with the added complexity of real-time intervention---our aim is to enable researchers, given a decent spiking model of the system of interest, to assess whether the type of control they desire is feasible and/or what configuration(s) would be most conducive to their goals.

2. The complexity of experimental interfaces means it's not always clear what a model would look like in a real experiment. Cleo can help anyone interested in observing or manipulating a model while taking into account the constraints present in real experiments. Because Cleo is built around the `Brian simulator <https://brian2.rtfd.io>`_, we especially hope this is helpful for existing Brian users who for whatever reason would like a convenient way to inject recorders (e.g., electrodes) or stimulators (e.g., optogenetics) into the core network simulation.

.. admonition:: What is closed-loop control? 
    
    In short, determining the inputs to deliver to a system from its outputs. In neuroscience terms, making the stimulation parameters a function of the data recorded in real time.

Structure and design
^^^^^^^^^^^^^^^^^^^^
Cleo wraps a spiking network simulator and allows for the injection of stimulators and/or recorders. The models used to emulate these devices are often non-trivial to implement or use in a flexible manner, so Cleo aims to make device injection and configuration as painless as possible, requiring minimal modification to the original network.

Cleo also orchestrates communication between the simulator and a user-configured :class:`~cleo.IOProcessor` object, modeling how experiment hardware takes samples, processes signals, and controls stimulation devices in real time.

For an explanation of why we choose to prioritize spiking network models and how we chose Brian as the underlying simulator, see :ref:`overview:design rationale`.

.. admonition:: Why closed-loop control in neuroscience?

    Fast, real-time, closed-loop control of neural activity enables intervention in processes that are too fast or unpredictable to control manually or with pre-defined stimulation, such as sensory information processing, motor planning, and oscillatory activity. 
    Closed-loop control in a *reactive* sense enables the experimenter to respond to discrete events of interest, such as the arrival of a traveling wave or sharp wave ripple, whereas *feedback* control deals with driving the system towards a desired point or along a desired state trajectory. 
    The latter has the effect of rejecting noise and disturbances, reducing variability across time and across trials, allowing the researcher to perform inference with less data and on a finer scale.
    Additionally, closed-loop control can compensate for model mismatch, allowing it to reach more complex targets where open-loop control based on imperfect models is bound to fail.

Installation
------------
Make sure you have Python >=3.7, then use pip: ``pip install cleosim``.

.. note:: 
    The name on PyPI is ``cleosim`` since ``cleo`` `was already taken <https://cleo.readthedocs.io/en/latest/>`_, but in code it is still used as ``import cleo``. 
    The other Cleo appears to actually be a fairly well developed package, so I'm sorry if you need to use it along with this Cleo in the same environment. 
    In that case, `there are workarounds <https://stackoverflow.com/a/55817170/6461032>`_.

Or, if you're a developer, `install poetry <https://python-poetry.org/docs/>`_ and run ``poetry install`` from the repository root.


Usage
-----

Brian network model
^^^^^^^^^^^^^^^^^^^
The starting point for using Cleo is a Brian spiking neural network model of the system of interest. For those new to Brian, the `docs <https://brian2.rtfd.io>`_ are a great resource. If you have a model built with another simulator or modeling language, you may be able to `import it to Brian via NeuroML <https://brian2tools.readthedocs.io/en/stable/user/nmlimport.html>`_.

Perhaps the biggest change you may have to make to an existing model to make it compatible with Cleo's optogenetics and electrode recording is to give the neurons of interest coordinates in space. See the :doc:`tutorials` or the :mod:`cleo.coords` module for more info.

You'll need your model in a Brian :class:`~brian2.core.network.Network` object before you move on. E.g.,::

    net = brian2.Network(...)

CLSimulator
^^^^^^^^^^^
Once you have a network model, you can construct a :class:`~cleo.CLSimulator` object::

    sim = cleo.CLSimulator(net)

The simulator object wraps the Brian network and coordinates device injection, processing input and output, and running the simulation.

Recording
^^^^^^^^^
Recording devices take measurements of the Brian network. Some extremely simple implementations (which do little more than wrap Brian monitors) are available in the :mod:`cleo.recorders` module. 

To use a :class:`~cleo.Recorder`, you must inject it into the simulator via :meth:`~cleo.CLSimulator.inject_recorder`::

    rec = MyRecorder('recorder_name', ...)  # note that all devices need a unique name
    sim.inject_recorder(rec, neuron_group1, neuron_group2, ...)  # can pass in additional arguments

The recorder will only record from the neuron groups specified on injection, allowing for such scenarios as singling out a cell type to record from.

Electrodes
""""""""""
Electrode recording is the main recording modality currently implemented in Cleo. See the :doc:`tutorials/electrodes` tutorial for more detail, but in brief, usages consists of:

#. Constructing a :class:`~cleo.ephys.Probe` object with coordinates at the desired contact locations

   * Convenience functions for generating shank probe coordinates exist. See :ref:`tutorials/electrodes:Specifying electrode coordinates`.

#. Specifying the signals to be recorded. Currently there are three implemented. See :ref:`tutorials/electrodes:Specifying signals to record`.

    * Multi-unit activity
    * Sorted spikes
    * TKLFP: Teleńczuk kernel approximation of LFP

#. Injection into the simulator


Stimulation
^^^^^^^^^^^
Stimulator devices manipulate the Brian network. Usage is similar to recorders::

    stim = MyStimulator('stimulator_name', ...)  # again, all devices need a unique name
    # again, specify neuron groups device will affect and any additional arguments needed
    sim.inject_stimulator(stim, neuron_group1, neuron_group2, ...)

As with recorders, you can inject stimulators per neuron group to produce a targeted effect.

Optogenetics
""""""""""""
Optogenetics is the main stimulator device currently implemented by Cleo. This take the form of an :class:`~cleo.opto.OptogeneticIntervention`, which, on injection, adds a light source at the specified location and transfects the neurons (via Brian "synapses" that deliver current according to an opsin model, leaving the neuron model equations untouched).

Out of the box you can access a four-state Markov model of channelrhodopsin-2 (ChR2) and parameters for a 473-nm blue optic fiber light source.::

    from cleo.opto import *
    opto = OptogeneticIntervention(
        name="...",
        opsin_model=FourStateModel(params=ChR2_four_state),
        light_model_params=default_blue,
        location=(0, 0, 0.5) * mm,
    )

Note, however, that Markov opsin dynamics models require target neurons to have membrane potentials in realistic ranges and an `Iopto` term defined in amperes. If you need to interface with a model without these features, you may want to use the simplified :class:`~cleo.opto.ProportionalCurrentModel`. You can find more details, including a comparison between the two model types, in the :ref:`optogenetics tutorial <tutorials/optogenetics:Appendix: alternative opsin and neuron models>`.
    
These model and parameter settings were designed to be flexible enough that an interested user should be able to imitate and replace them with other opsins, light sources, etc. See the :doc:`tutorials/optogenetics` tutorial for more detail.

IO Processor
^^^^^^^^^^^^
Just as in a real experiment where the experiment hardware must be connected to signal processing equipment and/or computers for recording and control, the :class:`~cleo.CLSimulator` must be connected to an :class:`~cleo.IOProcessor`::

    sim.set_io_processor(...)

If you are only recording, you may want to use the :class:`~cleo.ioproc.RecordOnlyProcessor`. Otherwise you will want to implement the :class:`~cleo.ioproc.LatencyIOProcessor`, which not only takes samples at the specified rate, but processes the data and delivers input to the network after a user-defined delay, emulating the latency inherent in real experiments. You define your processor by creating a subclass and defining the :meth:`~cleo.ioproc.LatencyIOProcessor.process` function::

    class MyProcessor(LatencyIOProcessor):

        def process(self, state_dict, sample_time_ms):
            # state_dict contains a {'recorder_name': value} dict of network
            foo = state_dict['foo_recorder']
            out = ... # do something with sampled spikes
            delay_ms = 3
            t_out_ms = sample_time_ms + delay_ms
            # output must be a {'stimulator_name': value} dict setting stimulator values
            return {'stim': out}, t_out_ms
    
    my_proc = MyProcessor(sample_period_ms=1)
    sim.set_io_processor(my_proc)

See :doc:`tutorials/on_off_ctrl` for a minimal working example or :doc:`tutorials/PI_ctrl` for more advanced features, including decomposing the processing into blocks with accompanying stochastic delay objects.

Running experiments
^^^^^^^^^^^^^^^^^^^
Use CLSimulator's :meth:`~cleo.CLSimulator.run` function with the desired duration::

    sim.run(500*ms, ...)  # kwargs are passed to Brian's run function

Use CLSimulator's :meth:`~cleo.CLSimulator.reset` function to restore the default state (right after initialization/injection) for the network and all devices. This could be useful for running a simulation multiple times under different conditions.

To facilitate access to data after the simulation, many classes offer a ``save_history`` option on construction. If true, that object will store relevant variables as attributes. For example,::

    sorted_spikes = cleo.ephys.SortedSpiking(...)
    ...
    sim.run(...)

    plt.plot(sorted_spikes.t_ms, sorted_spikes.i)


Design rationale
----------------

Why not prototype with more abstract models?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Cleo aims to be practical, and as such provides models at the level of abstraction corresponding to the variables the experimenter has available to manipulate. This means models of spatially defined, spiking neural networks.

Of course, neuroscience is studied at many spatial and temporal scales. While other projects may be better suited for larger segments of the brain and/or longer timescales (such as `HNN <https://elifesciences.org/articles/51214>`_ or BMTK's `PopNet <https://alleninstitute.github.io/bmtk/popnet.html>`_ or `FilterNet <https://alleninstitute.github.io/bmtk/filternet.html>`_), this project caters to finer-grained models because they can directly simulate the effects of alternate experimental configurations. For example, how would the model change when swapping one opsin for another, using multiple opsins simultaneously, or with heterogeneous expression? How does recording or stimulating one cell type vs. another affect the experiment? Would using a more sophisticated control algorithm be worth the extra compute time, and thus later stimulus delivery, compared to a simpler controller? 

Questions like these could be answered using an abstract dynamical system model of a neural circuit, but they would require the extra step of mapping the afore-mentioned details to a suitable abstraction---e.g., estimating a transfer function to model optogenetic stimulation for a given opsin and light configuration. Thus, we haven't emphasized these sorts of models so far in our development of Cleo, though they should be possible to implement in Brian if you are interested. For example, one could develop a Poisson linear dynamical system (PLDS), record spiking output, and configure stimulation to act directly on the system's latent state.

And just as experiment prototyping could be done on a more abstract level, it could also be done on an even more realistic level, which we did not deem necessary. That brings us to the next point...

Why Brian?
^^^^^^^^^^
Brian is a relatively new spiking neural network simulator written in Python. Here are some of its advantages:

* Flexibility: allowing (and requiring!) the user to define models mathematically rather than selecting from a pre-defined library of cell types and features. This enables us to define arbitrary models for recorders and stimulators and easily interface with the simulation
* Ease of use: it's all just Python
* Speed

`NEST <https://www.nest-simulator.org/>`_ is a popular alternative to Brian also strong in point neuron simulations. However, it appears to be less flexible, and thus harder to extend. `NEURON <https://www.neuron.yale.edu/neuron/>`_ is another popular alternative to Brian. Its main advantage is its first-class support of detailed, morphological, multi-compartment neurons. In fact, strong alternatives to Brian for this project were BioNet (`docs <https://alleninstitute.github.io/bmtk/bionet.html>`_, `paper <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0201630>`_) and NetPyNE (`docs <http://netpyne.org/index.html>`_, `paper <https://elifesciences.org/articles/44494>`_), which already offer a high-level interface to NEURON with extracellular potential recording. Optogenetics could be incorporated with `pre-existing .hoc code <https://github.com/ProjectPyRhO/PyRhO/blob/master/pyrho/NEURON/RhO4c.mod>`_, though the light model would need to be implemented. From brief examination of the `source code of BioNet <https://github.com/AllenInstitute/bmtk/blob/8c235eabbfa963a3fe163d6ba6e5ad67ca5ad7c3/bmtk/simulator/bionet/modules/sim_module.py#L44>`_, it appears that closed-loop stimulation would not be too difficult to add. It is unclear for NetPyNE.

In the end, we chose Brian since our priority was to model circuit/population-level dynamics over molecular/intra-neuron dynamics. Also, Brian does have support for multi-compartment neurons, albeit less fully featured, if that is needed.


Future development
------------------
Here are some features which are missing but could be useful to add:

* Better support for multiple opsins simultaneously. At present the user would have to include a separate variable for each new opsin current, which makes changing the number of different opsins inconvenient
* Support for multiple light sources affecting a single opsin transfection---whether the light sources have the same or different wavelengths
* Electrode microstimulation
* A more accurate LFP signal (only usable for morphological neurons) based on the volume conductor forward model as in `LFPy <https://lfpy.readthedocs.io/en/latest/index.html>`_ or `Vertex <https://github.com/haeste/Vertex_2>`_
* The `Mazzoni-Lindén LFP approximation <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004584>`_ for LIF point-neuron networks
* Imaging as a recording modality
