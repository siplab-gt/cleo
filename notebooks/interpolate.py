import pandas as pd
import numpy as np
import xarray as xr


num_files = int(input("Enter # of files: "))

xarrs = []
for i in range(num_files):
    path = input(f"Path to CSV #{i+1}: ")
    wl = int(path.split('_')[-1].split('.')[0])

    df = pd.read_csv(path, header=None)
    arr = df.values.reshape((256, 512, 256))
    da = xr.DataArray(
        arr,
        dims=("x", "y", "z"),
        coords={"x": np.arange(arr.shape[0]),
                "y": np.arange(arr.shape[1]),
                "z": np.arange(arr.shape[2])},
        attrs={"units": "mW/mm²"}

    ).expand_dims(wavelength=[wl])

    da = da.assign_coords(beam_size=0.0019)
    xarrs.append(da)

combined = xr.concat(xarrs, dim="wavelength").sortby("wavelength")
print(combined)
combined.to_netcdf("combined.nc")

w0 = float(input("Interp start wavelength: "))
w1 = float(input("Interp end wavelength: "))
wNum = int(input("# of wavelength points: "))

b0 = float(input("Interp start beam_size: "))
b1 = float(input("Interp end beam_size: "))
bNum= int(input("# of beam_size points: "))

waveList = np.linspace(w0, w1, wNum) 
beamList = np.linspace(b0, b1, bNum)  

if (bNum == 1):
    fluence_interp = combined.interp(wavelength= waveList, method="linear")
elif (wNum == 1):
    fluence_interp = combined.interp(beam_size = beamList, method="linear")
else:
    fluence_interp = combined.interp(wavelength = waveList, beam_size = beamList, method="linear")


print(fluence_interp)
fluence_interp.to_netcdf("fluence_interp.nc")
