"""Downloaded from https://zenodo.org/records/3866253#.Ydyk6GDMKUk
Data comes from the same source.

The following corrections were made to the computation:
- the calc_lfp() function correctly uses the data passed in, not always inh_cells
- va term switch to mm/ms to properly compute the delay
- s_e is used for excitatory cell amplitude rather than s_i
- amp_e and amp_e are unswitched

The following nonessential changes were made:
- added random seed to ensure reproducibility
- changed file path so spiking data is read from data/ directory
- changed plot save location and removed plt.show() command
- saved results to data/tklfp_demo.npz
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt

plt.switch_backend("agg")


# 1. read data in vectors

N = 5000  # nb of cells to consider
Ne = 4000  # nb of excitatory cells
Ni = 1000  # nb of inhibitory cells

tmin = 9000  # min time (to skip)
tmax = 1000  # max time

dtype = {"names": ["cellid", "time"], "formats": ["i4", "f8"]}
inh_cells = np.loadtxt("data/brunel_inh.txt", dtype=dtype)
exc_cells = np.loadtxt("data/brunel_exc.txt", dtype=dtype)

# adjust time and convert to ms
inh_cells["time"] = inh_cells["time"] * 1000 - tmin
exc_cells["time"] = exc_cells["time"] * 1000 - tmin
# for inhibitory cells, ids start from Ne
inh_cells["cellid"] += Ne

# 2. draw raster

Nstp = 10  # step cell to draw
tick_size = 5

fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

axes[0].plot(exc_cells[::Nstp]["time"], exc_cells[::Nstp]["cellid"], ".", ms=tick_size)
axes[0].plot(inh_cells[::Nstp]["time"], inh_cells[::Nstp]["cellid"], ".", ms=tick_size)

# 3. distribute cells in a 2D grid

xmax = 0.2  # size of the array (in mm)
ymax = 0.2

np.random.seed(1830)
X, Y = np.random.rand(2, N) * np.array([[xmax, ymax]]).T


# 4. calculate LFP
#
# Table of respective amplitudes:
# Layer   amp_i    amp_e
# deep    -2       -1.6
# soma    30       4.8
# sup     -12      2.4
# surf    3        -0.8
#

dt = 0.1  # time resolution
npts = int(tmax / dt)  # nb points in LFP vector

xe = xmax / 2
ye = ymax / 2  # coordinates of electrode

# va = 200  # axonal velocity (mm/sec)
va = 200 / 1000  # axonal velocity (mm/ms)
lambda_ = 0.2  # space constant (mm)
dur = 100  # total duration of LFP waveform
nlfp = int(dur / dt)  # nb of LFP pts
amp_e = 0.7  # uLFP amplitude for exc cells
amp_i = -3.4  # uLFP amplitude for inh cells
sig_i = 2.1  # std-dev of ihibition (in ms)
sig_e = 1.5 * sig_i  # std-dev for excitation

# amp_e = -0.16	# exc uLFP amplitude (deep layer)
# amp_i = -0.2	# inh uLFP amplitude (deep layer)

amp_e = 0.48  # exc uLFP amplitude (soma layer)
amp_i = 3  # inh uLFP amplitude (soma layer)

# amp_e = 0.24	# exc uLFP amplitude (superficial layer)
# amp_i = -1.2	# inh uLFP amplitude (superficial layer)

# amp_e = -0.08	# exc uLFP amplitude (surface)
# amp_i = 0.3	# inh uLFP amplitude (surface)

dist = np.sqrt((X - xe) ** 2 + (Y - ye) ** 2)  # distance to  electrode in mm
delay = 10.4 + dist / va  # delay to peak (in ms)
amp = np.exp(-dist / lambda_)
amp[:Ne] *= amp_e
amp[Ne:] *= amp_i

s_e = 2 * sig_e * sig_e
s_i = 2 * sig_i * sig_i


lfp_time = np.arange(npts) * dt


def f_temporal_kernel(t, tau):
    """function defining temporal part of the kernel"""
    return np.exp(-(t**2) / tau)


def calc_lfp(cells, s):
    """Calculate LFP from cells"""

    # this is a vectorised computation and as such it might be memory hungry
    # for long LFP series/large number of cells it may be more efficient to calculate it through looping

    spt = cells["time"]
    cid = cells["cellid"]
    kernel_contribs = amp[None, cid] * f_temporal_kernel(
        lfp_time[:, None] - delay[None, cid] - spt[None, :], s
    )
    lfp = kernel_contribs.sum(1)
    return lfp


lfp_inh = calc_lfp(inh_cells, s_i)
lfp_exc = calc_lfp(exc_cells, s_e)

total_lfp = lfp_inh + lfp_exc

axes[1].plot(lfp_time, total_lfp)
axes[1].set_xlabel("time, ms")
axes[1].set_xlim(0, tmax)

# prettify graph
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
plt.savefig("img/repl/telenczuk20_demo.pdf")
# plt.show()

np.savez(
    "data/telenczuk20_demo.npz",
    lfp=total_lfp,
    time=lfp_time,
    x=X,
    y=Y,
    t_spk=np.concatenate([exc_cells["time"], inh_cells["time"]]),
    i_spk=np.concatenate([exc_cells["cellid"], inh_cells["cellid"]]),
)
