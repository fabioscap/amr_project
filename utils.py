import numpy as np

def normalize(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))