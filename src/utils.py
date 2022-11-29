import torch
import yaml
from PIL import Image
from matplotlib import pyplot as plt
from mlflow.projects.utils import *
from mlflow.projects.utils import (
    _get_user,
    _get_git_commit
)
import io
import os

def get_mlflow_tags():
    tags = {MLFLOW_USER: _get_user()}
    source_version = _get_git_commit('.')

    if source_version is not None:
        tags[MLFLOW_GIT_COMMIT] = source_version

    if 'SLURM_JOB_ID' in os.environ:
        tags['JOB_ID'] = os.environ['SLURM_JOB_ID']
    elif 'LSB_JOBID' in os.environ:
        tags['JOB_ID'] = os.environ['LSB_JOBID']

    return tags