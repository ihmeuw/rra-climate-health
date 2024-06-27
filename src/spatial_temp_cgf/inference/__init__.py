from spatial_temp_cgf.inference.run_inference import (
    model_inference,
    model_inference_task,
)

RUNNERS = {
    "inference": model_inference,
}

TASK_RUNNERS = {
    "inference": model_inference_task,
}