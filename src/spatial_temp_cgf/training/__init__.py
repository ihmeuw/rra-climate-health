from spatial_temp_cgf.training.run_training import (
    model_training,
    model_training_task,
)

RUNNERS = {
    "training": model_training,
}

TASK_RUNNERS = {
    "training": model_training_task,
}
