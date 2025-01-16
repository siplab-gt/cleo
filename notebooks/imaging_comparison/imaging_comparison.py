import brian2 as b2
import matplotlib.pyplot as plt
from brian2 import np

import cleo
from cleo import imaging
from cleo.imaging import naomi, s2f

# for reproducibility
rng = np.random.default_rng(92)
np.random.seed(92)

cleo.utilities.style_plots_for_docs()
sensors = [
    naomi.gcamp6f_naomi(doub_exp_conv=False),
    naomi.jgcamp7f_naomi(doub_exp_conv=False),
]
s2f_sensors = [s2f.gcamp6f(), s2f.jgcamp7f()]


def create_neuron_group():
    """
    Create a neuron group with 100 neurons, each with a random location in a
    rectangular prism. Each neuron has a membrane potential that is governed
    by a leaky integrate-and-fire model with an additional optogenetic current.

    Returns:
        - ng: the neuron group
        - sim: the simulator
    """
    ng = b2.NeuronGroup(
        100,
        """dv/dt = (-(v - E_L) + Rm*Iopto) / tau_m : volt
        Iopto : amp""",
        threshold="v > -50*mV",
        reset="v=E_L",
        namespace={
            "tau_m": 20 * b2.ms,
            "Rm": 500 * b2.Mohm,
            "E_L": -70 * b2.mV,
        },
    )
    ng.v = -70 * b2.mV
    cleo.coords.assign_coords_rand_rect_prism(
        ng, [-75, 75], [-75, 75], [50, 150], unit=b2.um
    )
    sim = cleo.CLSimulator(b2.Network(ng))
    return ng, sim


def create_scope(sensor, method):
    """
    Create a scope with a focus depth of 100 um, an image width of 150 um, and
    the given sensor.

    Args:
        - sensor: the sensor to use

    Returns:
        - scope: the scope
    """
    return imaging.Scope(
        focus_depth=100 * b2.um,
        img_width=150 * b2.um,
        sensor=sensor,
        name=f"{sensor.name} {method} Scope",
    )


def inject_scope(scope, sim, ng, rho_rel_gen):
    """
    Inject the scope into the simulator and inject a random lognormal
    distribution of relative densities into the scope.

    Args:
        - scope: the scope to inject
        - sim: the simulator
        - ng: the neuron group
        - rho_rel_gen: the function to generate the relative
    """

    sim.inject(scope, ng, rho_rel_generator=rho_rel_gen)
    scope.inject_sensor_for_targets()


def main():
    ng, sim = create_neuron_group()

    rho_rel = rng.lognormal(0, 0.2, size=100)
    rho_rel_gen = lambda n: rho_rel[:n]

    scopes = []
    for sensor in sensors:
        scope = create_scope(sensor, "naomi")
        inject_scope(scope, sim, ng, rho_rel_gen)
        scopes.append(scope)
    for sensor in s2f_sensors:
        scope = create_scope(sensor, "s2f")
        inject_scope(scope, sim, ng, rho_rel_gen)
        scopes.append(scope)

    for scope in scopes:
        cleo.viz.plot(ng, devices=[scope])
        plt.savefig(f"setup_{scope.sensor.name}.png", dpi=300)

    sim.set_io_processor(cleo.ioproc.RecordOnlyProcessor(sample_period=1 * b2.ms))
    sim.run(1 * b2.second)

    sep = 0.5
    fig, axs = plt.subplots()
    colors = {"GCaMP6f": "#4daf4a", "jGCaMP7f": "#984ea3", "OGB-1": "#ff7f00"}
    for i, scope in enumerate(scopes):
        n2plot = len(scope.i_targets_for_neuron_group(ng))
        dFF = np.array(scope.dFF)[:, :n2plot]
        axs.plot(
            scope.t / b2.ms,
            dFF + sep * np.arange(n2plot),
            label=scope.sensor.name,
            lw=0.5,
            color=colors[scope.sensor.name],
        )
    axs.set_xlabel("Time (ms)")
    axs.set_ylabel("dFF")
    axs.set_yticks(sep * np.arange(n2plot))
    axs.set_yticklabels(range(1, n2plot + 1))
    handles, labels = axs.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # add legend so that it doesn't cover any of the plot
    axs.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncols=3,
    )

    plt.savefig("dFF.png", dpi=300)


if __name__ == "__main__":
    main()
