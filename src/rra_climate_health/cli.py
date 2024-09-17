import click

from rra_climate_health import data_prep, inference, training


@click.group()
def strun() -> None:
    """Entry point for running spatial-temporal CGF pipeline workflows."""


@click.group()
def sttask() -> None:
    """Entry point for running spatial-temporal CGF pipeline tasks."""


for module in [data_prep, training, inference]:
    runners = getattr(module, "RUNNERS", {})
    task_runners = getattr(module, "TASK_RUNNERS", {})

    for name, runner in runners.items():
        strun.add_command(runner, name=name)

    for name, task_runner in task_runners.items():
        sttask.add_command(task_runner, name=name)
