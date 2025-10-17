import numpy as np
import poselib

from utils.geometry import angle, rotation_angle


def random_3d_vector():
    phi = np.random.uniform(0, np.pi*2)
    costheta = np.random.uniform(-1,1)

    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x,y,z])

def qr_full(num_samples=1):
    z = np.random.randn(num_samples, 3, 3)
    q, r = np.linalg.qr(z)
    sign = 2 * (np.diagonal(r, axis1=-2, axis2=-1) >= 0) - 1
    rot = q
    rot *= sign[..., None, :]
    rot[:, 0, :] *= np.linalg.det(rot)[..., None]
    return rot

def test_homography():
    runs = 10000

    R_errs = []
    t_errs = []
    n_errs = []

    for _ in range(10000):
        t = random_3d_vector()
        n = random_3d_vector()
        R = qr_full()[0]
        d = np.random.uniform(1000, 10000)

        H = R - (np.dot(t[:, np.newaxis], n[np.newaxis, :]) /d)

        poses, ns = poselib.motion_from_homography(100 * H)
        if len(poses) == 2:
            # print(n, t)
            # print(ns[0], ns[1], poses[0].t, poses[1].t)


            R_errs.append(min([rotation_angle(pose.R.T @ R) for pose in poses]))
            t_errs.append(min([angle(pose.t / pose.t[0], t/t[0]) for pose in poses]))
            n_errs.append(min([angle(nn / nn[0], n/n[0]) for nn in ns]))
        else:
            R_errs.append(rotation_angle(poses[0].R.T @ R))

    print('Max R err: ', np.max(R_errs))
    print('Max t err: ', np.max(t_errs))
    print('Max n err: ', np.max(n_errs))

if __name__ == '__main__':
    test_homography()