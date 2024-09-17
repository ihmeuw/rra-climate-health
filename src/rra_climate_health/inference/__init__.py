from rra_climate_health.inference.run_inference import (
    forecast_scenarios_task,
    model_inference,
    model_inference_task,
)

RUNNERS = {
    "inference": model_inference,
}

TASK_RUNNERS = {
    "inference": model_inference_task,
    "forecast": forecast_scenarios_task,
}
