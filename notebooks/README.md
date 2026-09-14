# Cleo notebooks
This folder contains various Jupyter notebooks used in creating the Cleo manuscript.
Unlike those in `docs/tutorials/`, these are not guaranteed to work with the latest Cleo version.
As of the date of writing (June 2024), we will at least try to include the Cleo version and other environment information needed for reproducibility.

*Update March 2025*: [Marimo](https://marimo.io) is now recommended over Jupyter notebooks for future work.
This solves the reproducibility problem as long as "sandbox" mode is used, which keeps track of dependencies on a per-notebook basis and executes in an isolated environment:
```
marimo edit --sandbox opto_pr_fr_irr_exp_figs.py
```