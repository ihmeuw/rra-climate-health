from spatial_temp_cgf.data_prep.run_training_data_prep import (
    run_training_data_prep,
)
from spatial_temp_cgf.data_prep.run_inference_data_prep import (
    run_ldi_prep,
    run_ldi_prep_task,
)

RUNNERS = {
    "data_prep": run_training_data_prep,
    "ldi_prep": run_ldi_prep,
}

TASK_RUNNERS = {
    "data_prep": run_training_data_prep,
    "ldi_prep": run_ldi_prep_task,
}
