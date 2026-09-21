# Apptainer container

This replaces a local `.venv` with an Apptainer (Singularity) image,
`protac-permeability.sif`, containing Python 3.12 and every third-party
dependency from the project's [`pyproject.toml`](../pyproject.toml) /
[`uv.lock`](../uv.lock) (RDKit, scikit-learn, pandas, `py2opsin` + a JRE for
it, JupyterLab, etc. — the image is built with the `plotting` extra so
`notebooks/paper_figures.ipynb` works out of the box).

The image deliberately does **not** contain the project's own source code.
You bind-mount the live repository into the container at run time, so you
keep editing `protac_permeability/` on the host exactly as before and just
run it through the container's Python instead of a host `.venv`.

## Build

Run this from the **repository root** (the `%files` paths in the `.def` are
relative to the build's working directory):

```sh
apptainer build --fakeroot --ignore-fakeroot-command \
    apptainer/protac-permeability.sif \
    apptainer/protac-permeability.def
```

`--fakeroot --ignore-fakeroot-command` is needed on systems (such as
Berzelius login nodes) that have no `/etc/subuid`/`/etc/subgid` range
assigned to your user: `apt-get` normally drops privileges to the `_apt`
user while downloading packages, which requires a real subuid/subgid
mapping; the `.def` file works around this internally
(`APT::Sandbox::User=root`), but the two build flags are still required so
Apptainer runs the build at all instead of refusing outright. If your site
*has* configured subuid/gid ranges for you, a plain `--fakeroot` works too.

The build takes a few minutes and produces a self-contained `.sif` file of a
few hundred MB; it is not tracked by git (see `.gitignore`).

## Use

Bind-mount the repo and run modules from it (the `-m` form matters — it adds
the repo root to `sys.path`, which is how the existing scripts resolve
`from protac_permeability... import ...` without installing the package):

```sh
apptainer exec --bind "$PWD:$PWD" --pwd "$PWD" apptainer/protac-permeability.sif \
    python -m protac_permeability.permeability_surrogate.fit_surrogate \
    --data_path data/combined_protacs.csv \
    --save_dir models/test_model \
    --descriptor_cols MolecularWeight CharVol cLogP HeavyAtomCount RingCount \
        HydrogenBondAcceptorCount HydrogenBondDonorCount RotatableBondCount \
        TopologicalPolarSurfaceArea FractionCSP3 NumStereoCenters AllBonds \
        RingAtoms Halogens HeteroAtoms TNSA Flexibility
```

Or drop into a shell for interactive development/testing:

```sh
apptainer shell --bind "$PWD:$PWD" --pwd "$PWD" apptainer/protac-permeability.sif
```

Inside the shell, `python` is the baked venv's interpreter (via
`$PATH`, set in the image's `%environment`); edits you make to
`protac_permeability/` on the host are visible immediately since the
directory is bind-mounted, not copied.

`--bind "$PWD:$PWD" --pwd "$PWD"` is spelled out explicitly above because
whether Apptainer auto-binds arbitrary host paths (like `/proj` on
Berzelius) depends on site configuration; passing it explicitly works
regardless.

## Updating dependencies

The venv is baked into the image at build time, so it does not pick up
changes to `pyproject.toml`/`uv.lock` on its own. After changing
dependencies on the host as usual (`uv add ...` / `uv lock`), rebuild:

```sh
apptainer build --fakeroot --ignore-fakeroot-command --force \
    apptainer/protac-permeability.sif \
    apptainer/protac-permeability.def
```

To include the `mining` extra too (not installed by default — the mining
stage isn't part of the retraining/testing workflow above), edit the
`uv sync` line in `protac-permeability.def` to add `--extra mining`, then
rebuild.

## Jupyter notebooks

`paper_figures.ipynb` lives at `notebooks/paper_figures.ipynb`
and does `from protac_permeability... import ...` while also using paths
like `Path("../data/...")` — it expects its kernel's working directory to
be `notebooks/` (Jupyter's default for a notebook: cwd = the
notebook's own directory) while still being able to import
`protac_permeability` as a package. Since the image never installs
`protac_permeability` itself (see above), resolve the import by putting the
repo root on `PYTHONPATH` explicitly when launching the server:

```sh
apptainer exec --bind "$PWD:$PWD" --pwd "$PWD" --env "PYTHONPATH=$PWD" \
    apptainer/protac-permeability.sif \
    jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

This serves from the repo root, so both `paper_figures.ipynb` and any
scratch notebook you add elsewhere in the tree can find
`protac_permeability`. Jupyter prints a URL with a token
(`http://127.0.0.1:8888/lab?token=...`); on a remote machine like a
Berzelius login node, forward that port over SSH from your local machine
(`ssh -L 8888:localhost:8888 <user>@<host>`, or let VS Code's Remote-SSH
auto-forward it) and open the printed URL, or in VS Code use
**Jupyter: Specify Jupyter Server for Connections** and paste it in. The
kernel it hands out (`Python 3 (ipykernel)`) runs the image's baked
`/opt/venv` interpreter, and since the repo is bind-mounted rather than
copied in, edits to `protac_permeability/*.py` are picked up the next time
a notebook cell re-imports them (restart the kernel to force a clean
reload).
