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
_axis = 'axis'
_angle = 'angle'
freq = 50
fps = 30
step = 5

def write_json(keyframes, file):
    with open(file, 'w') as fp:
        json.dump(keyframes, fp)


def convert_euler_data(y):
    keyframes = {_LeftArm: [],
                 _LeftForeArm: [],
                 _LeftHand: [],
                 _RightArm: [],
                 _RightForeArm: [],
                 _RightHand: []}
    for i in range(0, y.shape[0], step):
        k = int(i*fps / freq)
        keyframes[_RightArm].append({_frame: k, _axis: _X, _angle: y[i, 0]})
        keyframes[_RightArm].append({_frame: k, _axis: _Y, _angle: y[i, 1]})
        keyframes[_RightArm].append({_frame: k, _axis: _Z, _angle: y[i, 2]})
        # keyframes[_RightForeArm].append({_frame: k, _axis: _X, _angle: y[i, 3]})
        # keyframes[_RightForeArm].append({_frame: k, _axis: _Y, _angle: y[i, 4]})
    return keyframes


if __name__ == '__main__':
    y = np.genfromtxt(sys.argv[1], delimiter=',')
    keyframes = convert_euler_data(y)
    write_json(keyframes, sys.argv[2])
