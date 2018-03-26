import numpy as np
import json
import sys

_LeftArm = 'LeftArm'
_LeftForeArm = 'LeftForeArm'
_LeftHand = 'LeftHand'
_RightArm = 'RightArm'
_RightForeArm = 'RightForeArm'
_RightHand = 'RightHand'
_X = 'X'
_Y = 'Y'
_Z = 'Z'
_frame = 'frame'
_quat = 'quat'
freq = 50
fps = 30
step = 5

def write_json(keyframes, file):
    with open(file, 'w') as fp:
        json.dump(keyframes, fp)


def convert_quat_data(y):
    keyframes = {_RightArm: [], _RightForeArm: []}
    for i in range(0, y.shape[0], step):
        keyframes[_RightArm].append({_frame: int(i*fps/freq), _quat: list(y[i, :4])})
        # keyframes[_RightForeArm].append({_frame: i*1.5, _quat: list(y[i, 4:8])})
    return keyframes


if __name__ == '__main__':
    y = np.genfromtxt(sys.argv[1], delimiter=',')
    keyframes = convert_quat_data(y)
    write_json(keyframes, sys.argv[2])
