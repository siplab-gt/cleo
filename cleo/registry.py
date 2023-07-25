from __future__ import annotations
from typing import Tuple

from attrs import define, field
from brian2 import NeuronGroup, Synapses, Subgroup
from brian2.units.allunits import meter2, nmeter, kgram, second, meter, joule

from cleo.base import CLSimulator
from cleo.coords import coords_from_ng


@define(repr=False)
class DeviceInteractionRegistry:
    """Facilitates the creation and maintenance of 'neurons' and 'synapses'
    implementing many-to-many light-opsin/indicator optogenetics"""

    sim: CLSimulator = field()

    subgroup_idx_for_light: dict["Light", slice] = field(factory=dict, init=False)

    light_source_ng: NeuronGroup = field(init=False, default=None)
    """Represents ALL light sources (multiple devices)"""

    ldds_for_ng: dict[NeuronGroup, set["LightDependentDevice"]] = field(
        factory=dict, init=False
    )

    lights_for_ng: dict[NeuronGroup, set["Light"]] = field(factory=dict, init=False)

    light_prop_syns: dict[Tuple["LightDependentDevice", NeuronGroup], Synapses] = field(
        factory=dict, init=False
    )

    connections: set[Tuple["Light", "LightDependentDevice", NeuronGroup]] = field(
        factory=set, init=False
    )

    light_prop_model = """
        T : 1
        epsilon : 1
        Ephoton : joule
        Irr_post = epsilon * T * Irr0_pre : watt/meter**2 (summed)
        phi_post = Irr_post / Ephoton : 1/second/meter**2 (summed)
    """

    def connect_light_to_ldd_for_ng(
        self, light: "Light", ldd: "LightDependentDevice", ng: NeuronGroup
    ):
        epsilon = ldd.epsilon(light.light_model.wavelength / nmeter)
        if epsilon == 0:
            return

        light_prop_syn = self._get_or_create_light_prop_syn(ldd, ng)
        if (light, ldd, ng) in self.connections:
            raise ValueError(f"{light} already connected to {ldd.name} for {ng.name}")

        i_source = self.subgroup_idx_for_light[light]
        light_prop_syn.epsilon[i_source, :] = epsilon
        light_prop_syn.T[i_source, :] = light.transmittance(coords_from_ng(ng)).ravel()
        # fmt: off
        # Ephoton = h*c/lambda
        light_prop_syn.Ephoton[i_source, :] = (
            6.63e-34 * meter2 * kgram / second
            * 2.998e8 * meter / second
            / light.light_model.wavelength
        )
        # fmt: on
        self.connections.add((light, ldd, ng))

    def _get_or_create_light_prop_syn(
        self, ldd: "LightDependentDevice", ng: NeuronGroup
    ) -> Synapses:
        if (ldd, ng) not in self.light_prop_syns:
            light_agg_ng = ldd.light_agg_ngs[ng.name]

            light_prop_syn = Synapses(
                self.light_source_ng,
                light_agg_ng,
                model=self.light_prop_model,
                name=f"light_prop_{ldd.name}_{ng.name}",
            )
            light_prop_syn.connect()
            # non-zero initialization to avoid nans from /0
            light_prop_syn.Ephoton = 1 * joule
            self.sim.network.add(light_prop_syn)
            self.light_prop_syns[(ldd, ng)] = light_prop_syn

        return self.light_prop_syns[(ldd, ng)]

    def register_ldd(self, ldd: "LightDependentDevice", ng: NeuronGroup):
        """Connects lights previously injected into this neuron group to this opsin"""
        if ng not in self.ldds_for_ng:
            self.ldds_for_ng[ng] = set()
        self.ldds_for_ng[ng].add(ldd)
        prev_injct_lights = self.lights_for_ng.get(ng, set())
        for light in prev_injct_lights:
            self.connect_light_to_ldd_for_ng(light, ldd, ng)

    def init_register_light(self, light: "Light"):
        if self.light_source_ng is not None:
            Irr0_prev = self.light_source_ng.Irr0
            n_prev = self.light_source_ng.N
            # need to remove the old light source from the network
            self.sim.network.remove(self.light_source_ng)
        else:
            Irr0_prev = []
            n_prev = 0

        # create new one
        self.light_source_ng = NeuronGroup(
            n_prev + light.n, "Irr0: watt/meter**2", name="light_source"
        )
        if n_prev > 0:
            self.light_source_ng[:n_prev].Irr0 = Irr0_prev
        self.sim.network.add(self.light_source_ng)
        self.subgroup_idx_for_light[light] = slice(n_prev, n_prev + light.n)

        # remove and replace light_prop_syns for previous connections
        for light_prop_syn in self.light_prop_syns.values():
            self.sim.network.remove(light_prop_syn)
        prev_cxns = self.connections.copy()
        self.connections.clear()
        self.light_prop_syns.clear()
        for light, ldd, ng in prev_cxns:
            self.connect_light_to_ldd_for_ng(light, ldd, ng)
        assert prev_cxns == self.connections

        return self.light_source_ng[n_prev : n_prev + light.n]

    def register_light(self, light: "Light", ng: NeuronGroup):
        """Connects light to opsins and indicators already injected into this neuron
        group"""
        # create new connections for this light
        if ng not in self.lights_for_ng:
            self.lights_for_ng[ng] = set()
        self.lights_for_ng[ng].add(light)
        prev_injct_ldds = self.ldds_for_ng.get(ng, set())
        for ldd in prev_injct_ldds:
            self.connect_light_to_ldd_for_ng(light, ldd, ng)

    def source_for_light(self, light: "Light") -> Subgroup:
        i = self.subgroup_idx_for_light[light]
        return self.light_source_ng[i]


registries: dict[CLSimulator, DeviceInteractionRegistry] = {}


def registry_for_sim(sim: CLSimulator) -> DeviceInteractionRegistry:
    """Returns the registry for the given simulator"""
    assert sim is not None
    if sim not in registries:
        registries[sim] = DeviceInteractionRegistry(sim)
    return registries[sim]
