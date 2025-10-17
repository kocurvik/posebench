import h5py
import numpy as np
from utils.geometry import *
from utils.misc import *
import poselib
import pycolmap
import datetime
import posebench
import cv2
from tqdm import tqdm
import argparse


# Compute metrics for relative pose estimation
# AUC for max(err_R,err_t) and avg/med for runtime
def compute_metrics(results, thresholds=[5.0, 10.0, 20.0]):
    methods = results.keys()
    metrics = {}
    for m in methods:
        max_err = [np.max((a, b)) for (a, b, _, _) in results[m]['errs']]
        metrics[m] = {}
        aucs = compute_auc(max_err, thresholds)
        for auc, t in zip(aucs, thresholds):
            metrics[m][f'AUC{int(t)}'] = auc

        aucs = compute_auc([x for (_, _, c, d) in results[m]['errs'] for x in (c, d)], thresholds)
        for auc, t in zip(aucs, thresholds):
            metrics[m][f'f AUC{int(t)}'] = auc

        metrics[m]['avg_rt'] = np.mean(results[m]['runtime'])
        metrics[m]['med_rt'] = np.median(results[m]['runtime'])

    return metrics


def eval_fundamental_focal_estimator(instance, F, info, F_rt, estimator='poselib'):
    opt = instance['opt']
    K1_gt = camera_dict_to_calib_matrix(instance['cam1'])
    pp1_gt = K1_gt[:2, 2]
    f1_gt = (K1_gt[0, 0] + K1_gt[1, 1]) / 2
    K2_gt = camera_dict_to_calib_matrix(instance['cam2'])
    pp2_gt = K2_gt[:2, 2]
    f2_gt = (K2_gt[0, 0] + K2_gt[1, 1]) / 2

    if instance['cam1']['height'] > 0.0 and instance['cam1']['width'] > 0.0:
        f1_prior = 1.2 * max(instance['cam1']['height'], instance['cam1']['width'])
    else:
        f1_prior = 2.4 * max(pp1_gt)

    if instance['cam2']['height'] > 0.0 and instance['cam2']['width'] > 0.0:
        f2_prior = 1.2 * max(instance['cam2']['height'], instance['cam2']['width'])
    else:
        f2_prior = 2.4 * max(pp2_gt)


    if instance['cam1']['height'] > 0.0 and instance['cam1']['width'] > 0.0:
        pp1 = np.array([instance['cam1']['width'] / 2, instance['cam1']['height'] / 2])
    else:
        pp1 = pp1_gt

    if instance['cam2']['height'] > 0.0 and instance['cam2']['width'] > 0.0:
        pp2 = np.array([instance['cam2']['width'] / 2, instance['cam2']['height'] / 2])
    else:
        pp2 = pp2_gt

    inl = info['inliers']

    if np.sum(inl) < 7:
        return [180.0, 180.0, 200.0, 200.0], F_rt

    if 'svd' in estimator:
        f_tt1 = datetime.datetime.now()
        camera1, camera2 = poselib.focals_from_fundamental(F, pp1, pp2)
        f1 = camera1.focal()
        f2 = camera2.focal()
        f_tt2 = datetime.datetime.now()

    # if 'direct' in estimator:
    #     f_tt1 = datetime.datetime.now()
    #     camera1, camera2 = poselib.focals_from_fundamental(F, pp1, pp2)
    #     f1 = camera1.focal()
    #     f2 = camera2.focal()
    #     f_tt2 = datetime.datetime.now()

    if 'hybrid' in estimator:
        f_tt1 = datetime.datetime.now()
        prior_cam1 = {'model': 'SIMPLE_PINHOLE', 'width': -1, 'height':-1, 'params': [f1_prior, pp1[0], pp1[1]]}
        prior_cam2 = {'model': 'SIMPLE_PINHOLE', 'width': -1, 'height':-1, 'params': [f2_prior, pp2[0], pp2[1]]}

        cam1, cam2, _ = poselib.focals_from_fundamental_iterative(F, prior_cam1, prior_cam2, max_iters=50, weights=np.array([1.0e-4, 1.0, 1.0e-4, 1.0]))
        f_tt2 = datetime.datetime.now()

        pp1 = np.array(cam1.params[1:])
        pp2 = np.array(cam2.params[1:])
        f1 = cam1.focal()
        f2 = cam2.focal()

        # if np.isnan(f1):
        #     print("f1 NaN")
        # if np.isnan(f2):
        #     print("f2 NaN")
        # if f1 == f1_prior:
        #     print("f1 prior")
        # if f2 == f2_prior:
        #     print("f2 prior")

    if 'gt' in estimator:
        f_tt1 = datetime.datetime.now()
        f_tt2 = datetime.datetime.now()
        f1 = f1_gt
        f2 = f2_gt

    if 'prior' in estimator:
        f_tt1 = datetime.datetime.now()
        f_tt2 = datetime.datetime.now()
        f1 = f1_prior
        f2 = f2_prior

    # if np.isnan(f1) or f1 < 0.0:
    #     f1 = f1_prior
    # if np.isnan(f2) or f2 < 0.0:
    #     f2 = f2_prior

    if np.isnan(f1) or np.isnan(f2):
        # return [180, 180, 100, 100], F_rt + (f_tt2 - f_tt1).total_seconds()
        return [180, 180, 100, 100], F_rt + (f_tt2 - f_tt1).total_seconds()

    K1 = np.array([[f1, 0.0, pp1[0]], [0.0, f1, pp1[1]], [0.0, 0.0, 1.0]])
    K2 = np.array([[f2, 0.0, pp2[0]], [0.0, f2, pp2[1]], [0.0, 0.0, 1.0]])
    E = K2.T @ F @ K1
    x1i = calibrate_pts(instance['x1'][inl], K1)
    x2i = calibrate_pts(instance['x2'][inl], K2)

    _, R, t, good = cv2.recoverPose(E, x1i, x2i)
    err_f1 = np.abs(f1_gt - f1) / f1_gt
    err_f2 = np.abs(f2_gt - f2) / f2_gt
    err_R = rotation_angle(instance['R'] @ R.T)
    err_t = angle(instance['t'], t)

    return [err_R, err_t, 100 * err_f1, 100 * err_f2], F_rt + (f_tt2 - f_tt1).total_seconds()


def eval_fundamental(instance, rfc=False):
    opt = instance['opt']
    K1_gt = camera_dict_to_calib_matrix(instance['cam1'])
    pp1_gt = K1_gt[:2, 2]
    K2_gt = camera_dict_to_calib_matrix(instance['cam2'])
    pp2_gt = K2_gt[:2, 2]

    if instance['cam1']['height'] > 0.0 and instance['cam1']['width'] > 0.0:
        pp1 = np.array([instance['cam1']['width'] / 2, instance['cam1']['height'] / 2])
    else:
        pp1 = pp1_gt

    if instance['cam2']['height'] > 0.0 and instance['cam2']['width'] > 0.0:

        pp2 = np.array([instance['cam2']['width'] / 2, instance['cam2']['height'] / 2])
    else:
        pp2 = pp2_gt

    opt['real_focal_check'] = rfc

    # x1 = instance['x1'] - pp1[np.newaxis, :]
    # x2 = instance['x2'] - pp2[np.newaxis, :]
    x1 = instance['x1']
    x2 = instance['x2']

    F_tt1 = datetime.datetime.now()
    F, info = poselib.estimate_fundamental(x1, x2, opt, {})
    F_tt2 = datetime.datetime.now()

    F_rt = (F_tt2 - F_tt1).total_seconds()

    return F, info, F_rt



def main(dataset_path='data/relative', force_opt={}, dataset_filter=[], method_filter=[]):
    datasets = [
        ('megadepth1500_sift', 2.0),
        ('megadepth1500_spsg', 2.0),
        ('megadepth1500_splg', 2.0),
        ('scannet1500_sift', 1.5),
        ('scannet1500_spsg', 1.5),
        ('imc_british_museum', 0.75),
        ('imc_london_bridge', 0.75),
        ('imc_piazza_san_marco', 0.75),
        ('imc_florence_cathedral_side', 0.75),
        ('imc_milan_cathedral', 0.75),
        ('imc_sagrada_familia', 0.75),
        ('imc_lincoln_memorial_statue', 0.75),
        ('imc_mount_rushmore', 0.75),
        ('imc_st_pauls_cathedral', 0.75)
    ]
    if len(dataset_filter) > 0:
        datasets = [(n, t) for (n, t) in datasets if substr_in_list(n, dataset_filter)]

    evaluators = {
        'F + SVD': lambda i, j, k, l: eval_fundamental_focal_estimator(i, j, k, l, estimator='svd'),
        'F(RFC) + SVD': lambda i, j, k, l: eval_fundamental_focal_estimator(i, j, k, l, estimator='rfc_svd'),
        # 'F + Direct': lambda i, j, k, l: eval_fundamental_focal_estimator(i, j, k, l, estimator='direct'),
        # 'F(RFC) + Direct': lambda i, j, k, l: eval_fundamental_focal_estimator(i, j, k, l, estimator='rfc_direct'),
        'F + Hybrid': lambda i, j, k, l: eval_fundamental_focal_estimator(i, j, k, l, estimator='hybrid'),
        'F(RFC) + Hybrid': lambda i, j, k, l: eval_fundamental_focal_estimator(i, j, k, l, estimator='rfc_hybrid'),
        # 'F + Prior': lambda i, j, k, l: eval_fundamental_focal_estimator(i, j, k, l, estimator='prior'),
        # 'F + GT': lambda i, j, k, l: eval_fundamental_focal_estimator(i, j, k, l, estimator='gt'),
    }
    if len(method_filter) > 0:
        evaluators = {k: v for (k, v) in evaluators.items() if substr_in_list(k, method_filter)}

    load = False

    metrics = {}
    full_results = {}
    for (dataset, threshold) in datasets:
        f = h5py.File(f'{dataset_path}/{dataset}.h5', 'r')
        if load:
            f_F = h5py.File(f'{dataset_path}/{dataset}_fundamental.h5', 'r')
        else:
            f_F = h5py.File(f'{dataset_path}/{dataset}_fundamental.h5', 'w')

        opt = {
            'max_reproj_error': threshold,
            'max_epipolar_error': threshold,
            'max_iterations': 1000,
            'min_iterations': 100,
            'success_prob': 0.9999
        }

        for k, v in force_opt.items():
            opt[k] = v

        results = {}
        for k in evaluators.keys():
            results[k] = {
                'errs': [],
                'runtime': []
            }

        for k, v in tqdm(f.items(), desc=dataset):
            instance = {
                'x1': v['x1'][:],
                'x2': v['x2'][:],
                'cam1': h5_to_camera_dict(v['camera1']),
                'cam2': h5_to_camera_dict(v['camera2']),
                'R': v['R'][:],
                't': v['t'][:],
                'threshold': threshold,
                'opt': opt
            }

            if load:
                F = np.array(f_F[k]['F'])
                info = {'inliers': np.array(f_F[k]['inliers'])[:]}
                F_rt = np.array(f_F[k]['F_rt'])[0]

                F_rfc = np.array(f_F[k]['F_rfc'])
                info_rfc = {'inliers': np.array(f_F[k]['inliers_rfc'])[:]}
                F_rt_rfc = np.array(f_F[k]['F_rt_rfc'])[0]
            else:
                F, info, F_rt = eval_fundamental(instance, rfc=False)
                F_rfc, info_rfc, F_rt_rfc = eval_fundamental(instance, rfc=True)

                grp = f_F.create_group(k)
                grp.create_dataset('F', data=F, shape=(3, 3))
                grp.create_dataset('F_rfc', data=F_rfc, shape=(3, 3))
                grp.create_dataset('inliers', data=info['inliers'], shape=(len(info['inliers'])), dtype=bool)
                grp.create_dataset('inliers_rfc', data=info_rfc['inliers'], shape=(len(info_rfc['inliers'])), dtype=bool)
                grp.create_dataset('F_rt', data=F_rt, shape=(1), dtype=float)
                grp.create_dataset('F_rt_rfc', data=F_rt_rfc, shape=(1), dtype=float)



            for name, fcn in evaluators.items():
                if 'RFC' in name:
                    errs, runtime = fcn(instance, F_rfc, info_rfc, F_rt_rfc)
                else:
                    errs, runtime = fcn(instance, F, info, F_rt)
                results[name]['errs'].append(np.array(errs))
                results[name]['runtime'].append(runtime)
        metrics[dataset] = compute_metrics(results)
        full_results[dataset] = results
    return metrics, full_results


if __name__ == '__main__':
    force_opt, method_filter, dataset_filter = posebench.parse_args()
    metrics, _ = main(force_opt=force_opt, method_filter=method_filter, dataset_filter=dataset_filter)
    avg_metrics = posebench.compute_average_metrics(metrics)
    posebench.print_metrics_per_dataset(metrics)
    print(20 * '*')
    print("Summary for all: ")
    posebench.print_metrics_per_method(avg_metrics)