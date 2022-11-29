
import torch.nn as nn
import yaml
from PIL import Image
from matplotlib import pyplot as plt
import io
import os


def load_state_dict_partially(model, state_dict):
    own_state = model.state_dict()

    for name, param in state_dict.items():
        if name not in own_state:
            print(f'Ignored parameter "{name}" on loading')
            continue
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            own_state[name].copy_(param)
        except RuntimeError:
            print(f'Ignored parameter "{name}" on loading')


def denormalize(x):
    return (x + 1) / 2


def plot_voxels(voxels, rot01=0, rot02=0, rot12=0):
    voxels = voxels[0]
    voxels[voxels >= 0.5] = 1
    voxels[voxels < 0.5] = 0
    voxels = voxels.rot90(rot01, (0, 1))
    voxels = voxels.rot90(rot02, (0, 2))
    voxels = voxels.rot90(rot12, (1, 2))
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_box_aspect((1, 1, 1))
    ax.voxels(voxels)

    buf = io.BytesIO()
    plt.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)

    plt.clf()
    plt.close()

    return img