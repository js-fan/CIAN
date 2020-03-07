import argparse
from lib.loader import VOCSegGroupLoader, VOCSegLoader
from lib.utils import *
from lib.models import *
import pydensecrf.densecrf as dcrf
from scripts.train_infer_segment import build_model


def generate_retrain_seeds(args, pid=-1):
    data_slice = None
    if pid >=0:
        gpus = args.gpus.split(',')
        data_slice = slice(pid, None, len(gpus))
        args.gpus = gpus[pid]
    args.batch_size = len(args.gpus.split(','))
    mod = build_model(args, False)

    with open(args.data_list, 'r') as f:
        data_names = [x.strip() for x in f.readlines()]
    if data_slice is not None:
        data_names = data_names[data_slice]
    image_src_list = [os.path.join(args.image_root, name+'.jpg') for name in data_names]
    pred_root = args.snapshot

    for name, img_src in zip(data_names, image_src_list):
        img = cv2.imread(img_src)[..., ::-1].copy()
        h, w = img.shape[:2]

        img_ = cv2.resize(img, (args.image_size, args.image_size))
        batch = mx.io.DataBatch(data=[mx.nd.array(img_[np.newaxis].transpose(0,3,1,2))])

        mod.forward(batch, is_train=False)
        probs = mod.get_outputs()[0].copy()

        if not args.no_mirror:
            batch2 = mx.io.DataBatch(data=[batch.data[0][:, :, :, ::-1]])
            mod.forward(batch2, is_train=False)
            probs_mirror = mod.get_outputs()[0][:, :, :, ::-1].copy()
            probs = (probs + probs_mirror) / 2
        prob = mx.nd.contrib.BilinearResize2D(probs, height=h, width=w).asnumpy()[0]
        
        pred = prob.argmax(axis=0).astype(np.uint8)

        d = dcrf.DenseCRF2D(w, h, prob.shape[0])
        u = - prob.reshape(prob.shape[0], -1)
        d.setUnaryEnergy(u)

        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=img, compat=10)
        #d.addPairwiseGaussian(sxy=1, compat=3)
        #d.addPairwiseBilateral(sxy=67, srgb=3, rgbim=img, compat=4)
            
        prob_crf = d.inference(10)
        prob_crf = np.array(prob_crf).reshape(-1, h, w)
        pred_crf = prob_crf.argmax(axis=0).astype(np.uint8)

        imwrite(os.path.join(pred_root, 'train_aug_pred', name+'.png'), pred)
        imwrite(os.path.join(pred_root, 'train_aug_pred_crf', name+'.png'), pred_crf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-root', type=str, required=True)
    parser.add_argument('--annotation-root', type=str, required=True)
    parser.add_argument('--label-root', type=str, default='')
    parser.add_argument('--train-list', type=str, default='data/VOC2012/train_aug.txt')
    parser.add_argument('--test-list', type=str, default='data/VOC2012/val.txt')
    parser.add_argument('--data-list', type=str, default='')
    parser.add_argument('--snapshot',   type=str, required=True)

    parser.add_argument('--model',      type=str, required=True)
    parser.add_argument('--pretrained', type=str, default='')

    # train
    parser.add_argument('--begin-epoch', type=int, default=0)
    parser.add_argument('--num-epoch',   type=int, default=20)
    parser.add_argument('--batch-size',  type=int, default=1)
    parser.add_argument('--image-size',  type=int, default=513)
    parser.add_argument('--num-cls',     type=int, default=21)
    parser.add_argument('--lr',          type=float, default=5e-4)

    parser.add_argument('--in-embed-type',  type=str, default='conv')
    parser.add_argument('--out-embed-type', type=str, default='convbn')
    parser.add_argument('--merge-type',     type=str, default='max')
    parser.add_argument('--group_size',     type=int, default=2)

    parser.add_argument('--log-frequency', type=int, default=50)
    parser.add_argument('--num-save',  type=int, default=5)
    parser.add_argument('--gpus', type=str, default='0', help='e.g. 0,1,2,3')

    # eval
    parser.add_argument('--pid', type=int, default=-1)
    parser.add_argument('--no-mirror', action='store_true')
    parser.add_argument('--no-ms', action='store_true')

    args = parser.parse_args()

    #
    if not args.data_list:
        args.data_list = args.train_list

    generate_retrain_seeds(args, args.pid)

