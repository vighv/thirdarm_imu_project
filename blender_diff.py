import numpy as np
import sys
import json
from transforms3d.euler import euler2mat, mat2euler

fps = 30
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


if __name__ == '__main__':
    # compute blender differences
    freq = int(sys.argv[3]) if len(sys.argv) > 3 else fps
    y = np.genfromtxt(sys.argv[1], delimiter=',')
    y = np.deg2rad(y)
    y = np.insert(y, 4, 0, 1)
    prev1 = np.eye(3)
    prev2 = np.eye(3)
    keyframes = {_RightArm: [],
                 _RightForeArm: []}
    step = max(1, int(round(freq / fps)))
    for i in range(0, y.shape[0], step):
        curr1 = euler2mat(y[i, 2], y[i, 1], y[i, 0])
        curr2 = euler2mat(y[i, 5], y[i, 4], y[i, 3])
        diff1 = curr1.dot(prev1.T)
        diff2 = curr2.dot(prev2.T)
        z1, y1, x1 = mat2euler(diff1)
        z2, y2, x2 = mat2euler(diff2)
        prev1 = curr1
        prev2 = curr2

        # translate into dictionary
        k = int(round(i * fps / freq))
        keyframes[_RightArm].append({_frame: k, _axis: _Z, _angle: z1})
        keyframes[_RightArm].append({_frame: k, _axis: _Y, _angle: y1})
        keyframes[_RightArm].append({_frame: k, _axis: _X, _angle: x1})
        keyframes[_RightForeArm].append({_frame: k, _axis: _Z, _angle: z2})
        keyframes[_RightForeArm].append({_frame: k, _axis: _Y, _angle: y2})
        keyframes[_RightForeArm].append({_frame: k, _axis: _X, _angle: x2})

    with open(sys.argv[2], 'w') as fp:
        json.dump(keyframes, fp)
