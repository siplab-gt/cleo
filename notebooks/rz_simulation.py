import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

#Change to actual source, target coordinates of the light
source = np.array([0, 0, 0])
target = np.array([0, 0, 1])

#Get theta and phi angles from source and target coordinates of light
dx, dy, dz = np.subtract(target, source)
r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
theta = np.arccos(dz / r)
phi = np.arctan2(dy, dx)

'''
print(theta)
print(phi)
'''
#Beam vector
bx = np.sin(theta) * np.cos(phi)
by = np.sin(theta) * np.sin(phi)
bz = np.cos(theta)
b = np.array([bx, by, bz])

fluence = xr.open_dataarray("fluence_interp.nc")
X, Y, Z = xr.broadcast(fluence.coords['x'], fluence.coords['y'], fluence.coords['z'])

#Computing axial (zarr) and lateral data
zarr = bx * X + by * Y + bz * Z
perpX = X - zarr * bx
perpY = Y - zarr * by
perpZ = Z - zarr * bz
rarr = np.sqrt(perpX ** 2 + perpY ** 2 + perpZ ** 2)
#print(rarr)


#Edges for r, z
r_max = rarr.max()
z_min = zarr.min()
z_max = zarr.max()

num_z_bins = 100
num_r_bins = 50
r_bins = np.linspace(0, r_max, num_r_bins + 1)
z_bins = np.linspace(z_min, z_max, num_z_bins + 1)

#Flattening to make same dimension for histogram2d
r_flat = rarr.values.ravel()
z_flat = zarr.values.ravel()

#Currently this is only for one wavelength/beam size combinations, defaults to first wavelength
#At the time of writing this the data is only for one beam size
fluence3d = fluence.isel(wavelength = 0)

fluence_flat = fluence3d.values.ravel()
print(r_flat.shape, z_flat.shape, fluence_flat.shape)

#Counts how many (x, y, z) points are in each (r, z) bin
counts, r_edges, z_edges = np.histogram2d(r_flat, z_flat, bins = [r_bins, z_bins])

#Sums the fluence values for each bin
sums, _, _ = np.histogram2d(r_flat, z_flat, bins = [r_bins, z_bins], weights = fluence_flat)

#Getting mean fluence in each bin
mean_fluence = np.divide(sums, counts, out = np.zeros_like(sums), where = (counts > 0))
mean_fluence[counts == 0] = np.nan 
r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

#Final result
fluence_cyl = xr.DataArray(mean_fluence, dims = ("r", "z"), coords = {"r": r_centers, "z": z_centers})
print(fluence_cyl)
fluence_cyl.to_netcdf("final_fluence.nc")

#Plotting
plt.pcolormesh(fluence_cyl['r'], fluence_cyl['z'], fluence_cyl.T, shading = 'auto')
plt.xlabel('r')
plt.ylabel('z')
plt.colorbar(label='mean fluence')
plt.show()
