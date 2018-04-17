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
_frame = 'frame'
_quat = 'quat'


def write_json(keyframes, file):
    with open(file, 'w') as fp:
        json.dump(keyframes, fp)


def add_euler_keyframe(keyframes, object, frame, axis, angle):
    keyframes[object].append({_frame: frame,
                              'axis': axis,
                              'angle': angle})


def convert_euler_data(y):
    keyframes = {_LeftArm: [],
                 _LeftForeArm: [],
                 _LeftHand: [],
                 _RightArm: [],
                 _RightForeArm: [],
                 _RightHand: []}
    for i in range(0, y.shape[0], 20):
        add_euler_keyframe(keyframes, _LeftArm, i, _X, y[i, 0])
        add_euler_keyframe(keyframes, _LeftArm, i, _Y, y[i, 1])
        add_euler_keyframe(keyframes, _LeftArm, i, _Z, y[i, 2])
        add_euler_keyframe(keyframes, _LeftForeArm, i, _X, y[i, 3])
        add_euler_keyframe(keyframes, _LeftForeArm, i, _Y, y[i, 4])
    return keyframes


def convert_quat_data(y):
    keyframes = {_RightArm: [], _RightForeArm: []}
    for i in range(0, y.shape[0], 2):
        keyframes[_RightArm].append({_frame: i*1.5, _quat: list(y[i, :4])})
        keyframes[_RightForeArm].append({_frame: i*1.5, _quat: list(y[i, 4:8])})
    return keyframes


if __name__ == '__main__':
    y = np.genfromtxt('data/Vicon_quat_data_23-Mar-2018.csv', delimiter=',')
    keyframes = convert_quat_data(y)
    write_json(keyframes, 'blender/anim-quat-23-Mar.json')
