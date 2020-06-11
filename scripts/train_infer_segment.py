import argparse
from lib.loader import VOCSegGroupLoader, VOCSegLoader
from lib.utils import *
from lib.models import *
import pydensecrf.densecrf as dcrf


def build_model(args, for_training=True):
    x_img = mx.sym.Variable('data')
    x_lbl = mx.sym.Variable('label')

    symbol_func = eval(args.model.lower() + \
            ('_CA' if for_training else '_SA') )
    if (not args.no_ms) and (not for_training):
        symbol_func = MultiScale([0.75, 1, 1.25])(symbol_func)

    x_pred = symbol_func(x_img, args.num_cls,
            use_global_stats_backbone=True,
            use_global_stats_affinity=not for_training,
            merge_self=True,
            in_embed_type=args.in_embed_type,
            out_embed_type=args.out_embed_type,
            merge_type=args.merge_type,
            group_size=args.group_size )
    
    if for_training:
        # cross entropy loss
        loss_list = [mx.sym.Custom(_pred, x_lbl, op_type='SegmentLoss') for _pred in x_pred]

        # completion loss
        x_cross = x_pred[1]
        loss_list += [mx.sym.Custom(_pred, x_cross, x_lbl, op_type='CompletionLoss') for _pred in x_pred]

        symbol = mx.sym.Group(loss_list)
    else:
        x_pred = [mx.sym.contrib.BilinearResize2D(_pred, 
                height=args.image_size, width=args.image_size) for _pred in x_pred]
        x_pred = x_pred[0] if len(x_pred) == 1 else sum(x_pred) / len(x_pred)
        symbol = x_pred

    # freeze backbone bn params
    aff_bn_names = []
    for suffix in ['_gamma', '_beta']:
        for aff_bn_name in ['bn_out', 'bn_embed_q', 'bn_embed_k', 'bn_embed_v']:
            aff_bn_names.append(aff_bn_name + suffix)

    fixed_param_names = []
    if for_training:
        fixed_param_names += [name for name in symbol.list_arguments() \
                if (name.endswith('_gamma') or name.endswith('_beta')) and (name not in aff_bn_names) ]

    # build model
    mod = mx.mod.Module(symbol, data_names=('data',),
            label_names=('label',) if for_training else None,
            context=[mx.gpu(int(gpu_id)) for gpu_id in setGPU(args.gpus).split(',')],
            fixed_param_names=fixed_param_names)
    label_size = (args.image_size - 1) // 8 +  1
    mod.bind(data_shapes=[('data', (args.batch_size, 3, args.image_size, args.image_size))],
            label_shapes=[('label', (args.batch_size, label_size, label_size))] if for_training else None )

    # load / initialize parameters
    if for_training:
        pretrained = args.pretrained
        if args.retrain:
            pretrained = os.path.join(args.snapshot.rsplit('_', 1)[0], '%s-%04d.params' % (args.model, args.num_epoch-1))
    else:
        pretrained = os.path.join(args.snapshot, '%s-%04d.params' % (args.model, args.num_epoch-1))
    assert os.path.exists(pretrained), pretrained
    info(None, "Using pretrained params: {}".format(pretrained), 'red')

    arg_params, aux_params = loadParams(pretrained)
    arg_params, aux_params = checkParams(mod, arg_params, aux_params, initializer=mx.init.Normal(0.01), auto_fix=for_training)

    if for_training and (not args.retrain):
        arg_params['bn_out_gamma'] *= 0

    mod.init_params(arg_params=arg_params, aux_params=aux_params)
    mod.init_optimizer(optimizer='sgd', optimizer_params={
        'learning_rate': args.lr,
        'momentum': 0.9,
        'wd': 5e-4},
        kvstore='device')

    return mod


def run_training(args):
    mod = build_model(args)

    loader = VOCSegGroupLoader(args.image_root, args.label_root, args.annotation_root,
            args.data_list, args.batch_size, args.group_size, len(mod._context), args.image_size,
            pad=False, shuffle=True, rand_scale=True, rand_mirror=True, rand_crop=True, downsample=8)

    saveParams = SaveParams(mod, args.snapshot, args.model, args.num_save)
    lrScheduler = LrScheduler('poly', args.lr, {'num_epoch': args.num_epoch, 'power': 0.9} )
    logger = getLogger(args.snapshot, args.model)
    summaryArgs(logger, vars(args), 'green')

    # train
    for n_epoch in range(args.begin_epoch, args.num_epoch):
        loader.reset()
        confmat = np.zeros((args.num_cls, args.num_cls), np.float32)
        loss = 0

        mod._optimizer.lr = lrScheduler.get(n_epoch)
        info(logger, "Learning rate: {}".format(mod._optimizer.lr), 'yellow')

        # monitor
        Timer.record()
        for n_batch, batch in enumerate(loader, 1):
            mod.forward_backward(batch)
            mod.update()

            if n_batch % args.log_frequency == 0:
                probs = mod.get_outputs()[0].as_in_context(mx.cpu())
                label = mx.nd.one_hot(batch.label[0], args.num_cls).transpose((0, 3, 1, 2))
                if probs.shape[2] != label.shape[2]:
                    label = mx.nd.contrib.BilinearResize2D(label, height=probs.shape[2], width=probs.shape[3])
                mask = label.sum(axis=1)
                _loss = -( (mx.nd.log((probs * label).sum(axis=1) + 1e-5) * mask).sum(axis=(1,2)) / \
                        mx.nd.maximum(mask.sum(axis=(1,2)), 1e-5) ).mean()

                loss_mom = (float(n_batch) - args.log_frequency) // n_batch
                loss = loss_mom * loss + (1 - loss_mom) * float(_loss.asnumpy())

                gt = label.argmax(axis=1).asnumpy().astype(np.int32)
                pred = probs.argmax(axis=1).asnumpy().astype(np.int32)
                assert gt.shape == pred.shape
                idx = label.max(axis=1).asnumpy() > 0.01
                confmat += np.bincount(gt[idx] * args.num_cls + pred[idx],
                        minlength=args.num_cls**2).reshape(args.num_cls, -1)
                iou = float((np.diag(confmat) / (confmat.sum(axis=0)+confmat.sum(axis=1)-np.diag(confmat)+1e-5)).mean())

                Timer.record()
                msg = "Epoch={}, Batch={}, miou={:.4f}, loss={:.4f}, speed={:.1f} b/s"
                msg = msg.format(n_epoch, n_batch, iou, loss, args.log_frequency/Timer.interval())
                info(logger, msg)

        saved_params = saveParams(n_epoch)
        info(logger, "Saved checkpoint:" + "\n  ".join(saved_params), 'green')


def run_infer(args, pid=-1):
    data_slice = None
    if pid >=0:
        gpus = args.gpus.split(',')
        data_slice = slice(pid, None, len(gpus))
        args.gpus = gpus[pid]
    args.batch_size = len(args.gpus.split(','))
    mod = build_model(args, False)

    loader = VOCSegLoader(args.image_root, None, args.data_list,
            args.batch_size, args.image_size, pad=True, shuffle=False,
            rand_scale=False, rand_mirror=False, rand_crop=False, data_slice=data_slice)
    pred_root = args.snapshot

    # inference
    for n_batch, batch in enumerate(loader, 1):
        image_src_list = loader.cache_image_src_list
        mod.forward(batch, is_train=False)
        probs = mod.get_outputs()[0].asnumpy()

        if not args.no_mirror:
            batch2 = mx.io.DataBatch(data=[batch.data[0][:, :, :, ::-1]])
            mod.forward(batch2, is_train=False)
            probs_mirror = mod.get_outputs()[0].asnumpy()[:, :, :, ::-1]
            probs = (probs + probs_mirror) / 2

        for img_src, prob in zip(image_src_list, probs):
            img = cv2.imread(img_src)[..., ::-1].copy()
            h, w = img.shape[:2]
            prob = prob[:, :h, :w]
            pred = prob.argmax(axis=0).astype(np.uint8)

            d = dcrf.DenseCRF2D(w, h, prob.shape[0])
            u = - prob.reshape(prob.shape[0], -1)
            d.setUnaryEnergy(u)

            # default CRF params
            d.addPairwiseGaussian(sxy=3, compat=3)
            d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=img, compat=10)
            
            prob_crf = d.inference(5)
            prob_crf = np.array(prob_crf).reshape(-1, h, w)
            pred_crf = prob_crf.argmax(axis=0).astype(np.uint8)

            name = os.path.basename(img_src).rsplit('.', 1)[0]
            imwrite(os.path.join(pred_root, 'pred', name+'.png'), pred)
            imwrite(os.path.join(pred_root, 'pred_crf', name+'.png'), pred_crf)


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
    parser.add_argument('--batch-size',  type=int, default=16)
    parser.add_argument('--image-size',  type=int, default=321)
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
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--pid', type=int, default=-1)
    parser.add_argument('--no-mirror', action='store_true')
    parser.add_argument('--no-ms', action='store_true')

    # retrain
    parser.add_argument('--retrain', action='store_true')

    args = parser.parse_args()

    #
    if not args.data_list:
        args.data_list = args.test_list if args.infer else args.train_list

    if args.retrain:
        args.snapshot += '_retrain'

    if args.infer:
        args.image_size = 513
        run_infer(args, args.pid)
    else:
        assert args.label_root
        if not args.retrain:
            assert args.pretrained
        run_training(args)

