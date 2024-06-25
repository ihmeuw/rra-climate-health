from spatial_temp_cgf.inference.inference_task import (
    model_inference,
    model_inference_task,
)

RUNNERS = {
    "inference": model_inference,
}

TASK_RUNNERS = {
    "inference": model_inference_task,
}
