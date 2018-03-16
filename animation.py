import numpy as np
import json

_LeftArm = 'LeftArm'
_LeftForeArm = 'LeftForeArm'
_LeftHand = 'LeftHand'
_RightArm = 'RightArm'
_RightForeArm = 'RightForeArm'
_RightHand = 'RightHand'
_X = 'X'
_Y = 'Y'
_Z = 'Z'


def add_keyframe(keyframes, object, frame, axis, angle):
    keyframes[object].append({'frame': frame,
                              'axis': axis,
                              'angle': angle})


def write_json(keyframes, file):
    with open(file, 'w') as fp:
        json.dump(keyframes, fp)


def convert_data(y):
    keyframes = {_LeftArm: [],
                 _LeftForeArm: [],
                 _LeftHand: [],
                 _RightArm: [],
                 _RightForeArm: [],
                 _RightHand: []}
    for i in range(0, y.shape[0], 30):
        add_keyframe(keyframes, _LeftArm, i, _X, y[i, 0])
        add_keyframe(keyframes, _LeftArm, i, _Y, y[i, 1])
        add_keyframe(keyframes, _LeftArm, i, _Z, y[i, 2])
        add_keyframe(keyframes, _LeftForeArm, i, _X, y[i, 3])
        add_keyframe(keyframes, _LeftForeArm, i, _Y, y[i, 4])
    return keyframes


if __name__ == '__main__':
    y = np.genfromtxt('data/vicon_proc01-Feb-2018.csv', delimiter=',')
    keyframes = convert_data(y)
    write_json(keyframes, 'data/anim_test.json')
