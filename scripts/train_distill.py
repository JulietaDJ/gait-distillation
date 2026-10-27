import sys
import os
import runpy
import numpy as np
import torch

sys.path.insert(0, "/opt3/data/julieta/OpenGait")
sys.path.insert(0, "/opt3/data/julieta/OpenGait/opengait")
sys.path.insert(0, "/opt3/data/julieta/OpenGait/lib")
sys.path.insert(0, "/opt3/data/julieta/gait-distillation/data")

os.chdir("/opt3/data/julieta/OpenGait")

import opengait.data.dataset
import opengait.data.collate_fn
import opengait.data
import opengait.modeling.base_model

from distillation_dataset import DistillationDataSet
from distillation_collate import DistillationCollateFn

opengait.data.dataset.DataSet = DistillationDataSet
if hasattr(opengait.data, 'DataSet'):
    opengait.data.DataSet = DistillationDataSet


opengait.data.collate_fn.CollateFn = DistillationCollateFn
if hasattr(opengait.data, 'CollateFn'):
    opengait.data.CollateFn = DistillationCollateFn
if hasattr(opengait.modeling.base_model, 'CollateFn'):
    opengait.modeling.base_model.CollateFn = DistillationCollateFn

if __name__ == "__main__":
    runpy.run_path("/opt3/data/julieta/OpenGait/opengait/main.py", run_name="__main__")

