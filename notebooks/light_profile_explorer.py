import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md("""
    # OptogenSIM profile explorer

    Interactive 2D cross-sections of the OptogenSIM Monte Carlo light model.
    Drag the sliders to set a wavelength and beam radius and see how light
    propagates through gray matter. The heatmap shows log₁₀ transmittance in
    the (r, z) plane, mirrored across the beam axis. z is depth from the
    source; r is radial distance.

    *Note: values between simulated grid points (wavelengths every 30 nm;
    beam radii 10, 20, 100, 200, 400, 800 µm) are linearly interpolated.
    """)
    return


@app.cell
def _():
    import numpy as np
    import plotly.graph_objects as go
    from cleo.light import OptogenSIM
    from brian2 import nmeter, um

    return OptogenSIM, go, nmeter, np, um


@app.cell
def _(OptogenSIM):
    _m = OptogenSIM()  # default instance, just to read the packaged data grid
    wl_vals = _m.data.wavelength.values
    bs_vals = _m.data.beam_size.values
    return bs_vals, wl_vals


@app.cell
def _(bs_vals, mo, wl_vals):
    wavelength = mo.ui.slider(
        start=float(wl_vals.min()),
        stop=float(wl_vals.max()),
        step=5,
        value=470.0,
        label="Wavelength (nm)",
        show_value=True,
    )
    beam = mo.ui.slider(
        start=float(bs_vals.min()),
        stop=float(bs_vals.max()),
        step=10,
        value=200.0,
        label="Beam radius (µm)",
        show_value=True,
    )
    mo.hstack([wavelength, beam], justify="start")
    return beam, wavelength


@app.cell
def _(OptogenSIM, beam, go, nmeter, np, um, wavelength):
    model = OptogenSIM(
        wavelength=wavelength.value * nmeter,
        beam_radius=beam.value * um,
    )
    sl = model._rz_slice  # interpolated to wavelength/beam; z already re-zeroed to source

    r = sl.r.values * 10.0        # cm -> mm
    z_rel = sl.z.values * 10.0    # cm -> mm (already source-relative)

    T = sl.values
    T = T / np.nanmax(T)
    logT = np.log10(np.clip(T, 1e-4, None))

    r_full = np.concatenate([-r[::-1], r])
    logT_full = np.vstack([logT[::-1, :], logT])

    fig = go.Figure(
        data=go.Heatmap(
            x=z_rel,
            y=r_full,
            z=logT_full,
            colorscale="Viridis",
            zmin=-4, zmax=0,
            colorbar=dict(title="log₁₀ T/peak"),
        )
    )
    fig.update_layout(
        title=f"{wavelength.value:.0f} nm, {beam.value:.0f} µm beam radius",
        xaxis_title="z, depth from source (mm)",
        yaxis_title="r, radial distance (mm)",
        width=750, height=450,
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    fig.update_xaxes(range=[float(z_rel.min()), min(float(z_rel.max()), 3)])
    fig
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
