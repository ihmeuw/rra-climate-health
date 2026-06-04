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
