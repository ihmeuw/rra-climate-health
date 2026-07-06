# RRA Climate Health

[Documentation](https://ihmeuw.github.io/rra-climate-health/)


A collection of geospatial models examining the relationship between climate variables,
socio-demographic indicators, and health outcomes.

## Setting up a development environment

This project uses [pixi](https://pixi.sh) to manage both the Python and the R
sides of the environment in a single lockfile. Pixi pulls R itself (and
`r-scam`, `r-lme4`, `r-lmertest`, `r-emmeans`, `r-mgcv`) from conda-forge so
that `rpy2` is ABI-compatible with the R it links against at runtime — this
is what previously made the install brittle across machines.

* Install pixi (once per machine):

    ```sh
    curl -fsSL https://pixi.sh/install.sh | sh
    ```

* Clone this repository.

* Install the runtime environment:

    ```sh
    pixi install
    ```

* Drop into a shell with the environment activated:

    ```sh
    pixi shell
    ```

  Or run a single command in the env without activating:

    ```sh
    pixi run strun ...
    ```

### Submitting jobs to the IHME cluster

`jobmon` lives in the optional `cluster` feature because it comes from the
IHME artifactory (which requires being on the IHME network — VPN or
in-office). Add it on top of the default env:

```sh
pixi install -e cluster      # default + jobmon
pixi shell -e cluster
```

### Development tools

Linters, type checking, tests, and docs live in the optional `dev` feature.

```sh
pixi install -e dev          # default + dev tooling
pixi shell -e dev
pixi run -e dev pytest

# Both at once (typical for hands-on work on the cluster):
pixi install -e cluster-dev
```

### Updating the lockfile

Edit `[tool.pixi.*]` in `pyproject.toml`, then:

```sh
pixi install             # re-solves and updates pixi.lock
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

### Pre-commit

Pre-commit hooks run all the auto-formatting (`ruff format`), linters (e.g. `ruff` and `mypy`), and other quality
 checks to make sure the changeset is in good shape before a commit/push happens.

You can install the hooks with (runs for each commit):

```sh
pixi run -e dev pre-commit install
```

Or if you want them to run only for each push:

```sh
pixi run -e dev pre-commit install -t pre-push
```

Or if you want e.g. want to run all checks manually for all files:

```sh
pixi run -e dev pre-commit run --all-files
```

---
