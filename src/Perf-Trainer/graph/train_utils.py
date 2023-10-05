import os
import datetime

import numpy as np
import cv2
import torch
import torchvision

def save_checkpoint(state, savepath, flag=True):
    """Save for general purpose (e.g., resume training)"""
    if not os.path.isdir(savepath):
        os.makedirs(savepath, 0o777)
    # timestamp = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    if flag:
        filename = os.path.join(savepath, "best_ckpt.pth.tar")
    else:
        filename = os.path.join(savepath, "newest_ckpt.pth.tar")
    torch.save(state, filename)


def load_checkpoint(savepath, flag=True):
    """Load for general purpose (e.g., resume training)"""
    if flag:
        filename = os.path.join(savepath, "best_ckpt.pth.tar")
    else:
        filename = os.path.join(savepath, "newest_ckpt.pth.tar")
    if not os.path.isfile(filename):
        return None
    state = torch.load(filename)
    return state

def log_scalar_list(logger, name, g_step, scalar_list):
    for scalar in scalar_list:
        logger.add_scalar(name, scalar, g_step)
        g_step+=1
    return g_step


