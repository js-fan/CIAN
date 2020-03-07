import argparse
import numpy as np
import multiprocessing as mp
import cv2
import os

def run_eval(data_list, pred_root, gt_root, num_cls):
    def compute_confusion_matrix(names, label_root, pred_root, num_cls, num_threads=16, arr_=None):
        if num_threads == 1:
            mat = np.zeros((num_cls, num_cls), np.float32)
            for name in names:
                gt = cv2.imread(os.path.join(label_root, name+'.png'), 0).astype(np.int32)
                pred = cv2.imread(os.path.join(pred_root, name+'.png'), 0).astype(np.int32)
                if gt.shape != pred.shape:
                    info(None, "NAME {}, gt.shape != pred.shape: [{} vs. {}]".format(name, gt.shape, pred.shape), 'red')
                    continue

                valid = gt < num_cls
                mat += np.bincount(gt[valid] * num_cls + pred[valid], minlength=num_cls**2).reshape(num_cls, -1)

            if arr_ is not None:
                arr_mat = np.frombuffer(arr_.get_obj(), np.float32)
                arr_mat += mat.ravel()
            return mat
        else:
            workload = np.full((num_threads,), len(names)//num_threads, np.int32)
            if workload.sum() < len(names):
                workload[:(len(names) - workload.sum())] += 1
            workload = np.cumsum(np.hstack([0, workload]))
            
            names_split = [names[i:j] for i, j in zip(workload[:-1], workload[1:])]

            arr_ = mp.Array('f', np.zeros((num_cls * num_cls,), np.float32))
            mat = np.frombuffer(arr_.get_obj(), np.float32).reshape(num_cls, -1)

            jobs = [mp.Process(target=compute_confusion_matrix, args=(_names, label_root, pred_root, num_cls, 1, arr_)) \
                    for _names in names_split]
            res = [job.start() for job in jobs]
            res = [job.join() for job in jobs]
            return mat.copy()

    def compute_eval_results(confmat):
        iou = np.diag(confmat) / np.maximum(confmat.sum(axis=0) + confmat.sum(axis=1) - np.diag(confmat), 1e-10)
        return iou

    # 
    with open(data_list) as f:
        names = [x.strip() for x in f.readlines()]

    confmat = compute_confusion_matrix(names, gt_root, pred_root, num_cls)
    iou = compute_eval_results(confmat)

    msg = "mIOU: {}\n{}\n\n".format(iou.mean(), iou)
    print(msg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-cls', type=int, default=21)
    parser.add_argument('--data-list', type=str, default='./data/VOC2012/val.txt')
    parser.add_argument('--prediction-root',  type=str, required=True)
    parser.add_argument('--groundtruth-root', type=str, required=True)

    args = parser.parse_args()
    run_eval(args.data_list, args.prediction_root, args.groundtruth_root, args.num_cls)

