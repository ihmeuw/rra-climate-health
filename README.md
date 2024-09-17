# RRA Climate Health

[Documentation](https://ihmeuw.github.io/rra-climate-health/)


A collection of geospatial models examining the relationship between climate variables,
socio-demographic indicators, and health outcomes.

## Setting up a development environment

* Clone this repository
* Create a conda environment with python and the R dependencies for the model.

    ```sh
    conda create -n cgf -c conda-forge python=3.12 r r-base r-lmertest r-emmeans
    ```

* Activate the conda environment

    ```sh
    conda activate cgf
    ```

* Use `pip` to install `poetry` in the conda environment

    ```sh
    pip install poetry
    ```

* Install the dependencies

    ```sh
    poetry install
    ```

### Pre-commit

Pre-commit hooks run all the auto-formatting (`ruff format`), linters (e.g. `ruff` and `mypy`), and other quality
 checks to make sure the changeset is in good shape before a commit/push happens.

You can install the hooks with (runs for each commit):

```sh
pre-commit install
```

Or if you want them to run only for each push:

```sh
pre-commit install -t pre-push
```

Or if you want e.g. want to run all checks manually for all files:

```sh
pre-commit run --all-files
```

---
