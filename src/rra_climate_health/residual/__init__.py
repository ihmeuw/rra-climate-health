from rra_climate_health.residual.run_residual import (
    residual_task,
    run_residual,
)

RUNNERS = {
    "residual": run_residual,
}

TASK_RUNNERS = {
    "residual": residual_task,
}
