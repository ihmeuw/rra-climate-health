# RRA Climate Health

[Documentation](https://ihmeuw.github.io/rra-climate-health/)


A collection of geospatial models examining the relationship between climate variables,
socio-demographic indicators, and health outcomes.

## Quick start

**If you are working on the IHME cluster, this is what you want:**

```sh
# 1. Install pixi (once per machine)
curl -fsSL https://pixi.sh/install.sh | sh

# 2. Clone this repository, then from inside it:
pixi install -e cluster-dev

# 3. Drop into a shell with that environment activated
pixi shell -e cluster-dev
```

Everything (Python, R, `jobmon`, linters, tests) is now on your `PATH`, so
`strun`, `sttask`, `pytest`, `python` etc. work directly.

> **Note:** plain `pixi install` builds **only** the `default` environment,
> which has *no* `jobmon` and *no* dev tooling. It does not install the other
> environments. Almost everyone working on this repo needs `-e cluster-dev` —
> pass the `-e` flag or you will be missing `jobmon` when you try to launch
> jobs. See [Environments](#environments) below.

Step 2 must be run **on the IHME network** (in-office or on VPN), because
`jobmon` comes from the IHME artifactory. Everything else works anywhere.

## Running the pipeline

Four steps, launched with `strun` (submits a `jobmon` workflow to SLURM) or run
directly with `sttask` (one unit of work in the current process):

```sh
strun training specifications/stunting.yaml -q long.q       # -> model version
strun inference -m stunting -t <model-version> \
    -c all -y all -s all -a all -d 100                      # -> results version, then forecast
    # add --save-rasters for 2023/2100 rasters + raster diff maps
strun residual -m stunting -r <results-version>             # -> final draws, SEVs, plots
```

You can also run `strun training --help` to understand the command options.

See [Running the pipeline](https://ihmeuw.github.io/rra-climate-health/running/)
for what each step consumes and produces, how versions chain together, and how
to pick a run back up after a failure, and
[Current Versions](https://ihmeuw.github.io/rra-climate-health/versions/) for
the model and results versions in use per measure.

## Environments

This project uses [pixi](https://pixi.sh) to manage both the Python and the R
sides of the environment in a single lockfile (`pixi.lock`, checked in). There
are four environments, built from three optional feature sets:

| Environment   | What you get                                  | Use it when                                          | Needs IHME network |
| ------------- | --------------------------------------------- | ---------------------------------------------------- | ------------------ |
| `default`     | Python + R + the modelling stack              | Running models on a laptop / off-network CI          | no                 |
| `dev`         | `default` + ruff, mypy, pytest, mkdocs        | Linting/testing without cluster submission           | no                 |
| `cluster`     | `default` + `jobmon`                          | Submitting SLURM jobs, no code changes               | yes                |
| `cluster-dev` | `default` + `dev` + `cluster`                 | **The normal choice** — hands-on work on the cluster | yes                |

Useful things to know about pixi's environment handling:

* `pixi install` installs **one** environment at a time. With no `-e` flag it
  installs `default`. Use `-e <name>` to pick another, or `--all` to build all
  four at once (this needs IHME network access, because of the `cluster` envs).
* `pixi run` and `pixi shell` auto-install the environment if it is missing, so
  `pixi install` is really just a way to pre-warm it.
* The `-e` flag applies per command, so it is easiest to activate once per
  terminal with `pixi shell -e cluster-dev` and then work normally.

```sh
pixi shell -e cluster-dev              # activate for this terminal
pixi run -e cluster-dev strun ...      # or run a single command, no activation
pixi run -e cluster-dev pytest         # run the tests
pixi list -e cluster-dev               # what actually got installed
```


`jobmon` is kept out of the `default` environment on purpose: it comes from the
IHME artifactory, so bundling it into `default` would make a plain
`pixi install` fail for anyone off the IHME network (including public GitHub
Actions runners).

## Development

### Pre-commit

Pre-commit hooks run all the auto-formatting (`ruff format`), linters (e.g.
`ruff` and `mypy`), and other quality checks to make sure the changeset is in
good shape before a commit/push happens.

Install the hooks so they run on each commit:

```sh
pixi run -e cluster-dev pre-commit install
```

Or so they run only on each push:

```sh
pixi run -e cluster-dev pre-commit install -t pre-push
```

Or run all checks manually against all files:

```sh
pixi run -e cluster-dev pre-commit run --all-files
```

(`-e dev` works just as well for any of these if you don't need `jobmon`.)

### Updating the lockfile

Edit `[tool.pixi.*]` in `pyproject.toml`, then re-solve and install:

```sh
pixi install -e cluster-dev    # re-solves and updates pixi.lock
```

Commit both `pyproject.toml` and `pixi.lock` together.

### Updating jobmon

`jobmon` needs special handling because of how it is published. The
`jobmon-installer-ihme` meta-package lives *only* on the IHME artifactory and
pins exact versions of its siblings (`jobmon-client`, `jobmon-core`,
`jobmon-slurm`, `slurm-rest`). Those siblings also exist on public PyPI, but
with unrelated version lineages (e.g. public `jobmon-slurm 2.0.0` is a
different project from the IHME `1.15.0`), so their names are effectively
shadowed.

Because of this, a bare `pixi update jobmon-installer-ihme -e cluster` will
**not** advance an open (`"*"`) requirement — pixi reports "Lock-file was
already up-to-date" and stays on the locked version even when a newer one is
available on the artifactory. This is not a cache problem and not a dependency
conflict; pixi/uv simply will not bump an unconstrained requirement for a
package that only lives on the secondary (artifactory) index.

To move jobmon to a new version:

1. Set a version constraint in `pyproject.toml` under
   `[tool.pixi.feature.cluster.pypi-dependencies]` that excludes the currently
   locked version — either raise the floor (`jobmon-installer-ihme = ">=10.12.2"`)
   or pin exactly (`== 10.12.2`).

2. Re-solve and install (this must be run on the IHME network so the
   artifactory is reachable):

    ```sh
    pixi update jobmon-installer-ihme -e cluster
    ```

   This rewrites `pixi.lock` for both the `cluster` and `cluster-dev`
   environments (they share a solve group) and installs into `cluster`.

3. Sync the `cluster-dev` environment too:

    ```sh
    pixi install -e cluster-dev
    ```

4. Verify the versions landed and jobmon still imports:

    ```sh
    pixi list -e cluster | grep -i jobmon
    pixi run -e cluster-dev python -c "import jobmon.client.tool; print('ok')"
    ```

Commit both `pyproject.toml` and `pixi.lock`. Note that a plain `pixi update`
likely will not auto-bump past the constraint next time either, so raise the
floor (or the pin) again when you want a newer jobmon.

---
