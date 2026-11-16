# %%
import pickle
import brian2.only as b2
import numpy as np

# %%
ng = b2.NeuronGroup(10, "dv/dt = -v / (10*ms) : 1", threshold="v>1", name="neurons")
ng.v = np.random.rand(10)
syn = b2.Synapses(ng, ng, "w : 1", on_pre="v_post += w", name="synapses")
syn.connect("i != j")
syn.w = "rand()"
st_mon = b2.StateMonitor(ng, "v", record=True, name="st_mon")
spk_mon = b2.SpikeMonitor(ng, name="spk_mon")
net = b2.Network(ng, syn, st_mon, spk_mon)
net.store(filename="netstore.pkl")

# %%
ng = b2.NeuronGroup(10, "dv/dt = -v / (10*ms) : 1", threshold="v>1", name="neurons")
ng.v = np.random.rand(10)
syn = b2.Synapses(ng, ng, "w : 1", on_pre="v_post += w", name="synapses")
syn.connect("i != j")
# syn.w = "rand()"
st_mon = b2.StateMonitor(ng, "v", record=True, name="st_mon")
spk_mon = b2.SpikeMonitor(ng, name="spk_mon")
net2 = b2.Network(ng, syn, st_mon, spk_mon)
net2.restore(filename="netstore.pkl")
# %%
# net.__dir__
import dill

b2.prefs.codegen.target = "numpy"
ng = b2.NeuronGroup(
    10,
    "dv/dt = -v / (10*ms) : volt \n du/dt = -u / ms : 1",
    name="neurons",
)
ng.v = 1 * b2.mvolt
st_mon = b2.StateMonitor(ng, ["v", "u"], record=True, name="st_mon")
net = b2.Network(ng, st_mon)
net.run(2 * b2.ms)
# st_mon.get_states()
# net.store(filename="netstore.pkl")

with open("net.pkl", "wb") as f:
    dill.dump(net, f)

# %%
import pickle

with open("netstore.pkl", "rb") as f:
    store = pickle.load(f)
store

# %%
np.zeros_like(ng.v)
