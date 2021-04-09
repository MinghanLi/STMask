from datasets import *
from utils.functions import MovingAverage, SavePath, ProgressBar
from utils.logger import Log
from utils import timer
from layers.modules import MultiBoxLoss
from STMask import STMask
import os
import time
import math
import torch
from datasets import get_dataset, prepare_data
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import argparse
import datetime

# Oof
import eval as eval_script


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Yolact Training Script')
parser.add_argument('--batch_size', default=6, type=int,
                    help='Batch size for training')
parser.add_argument('--eval_batch_size', default=1, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from. If this is "interrupt"' \
                         ', the model will resume training from the interrupt file.')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter. If this is -1, the iteration will be' \
                         'determined from the file name.')
parser.add_argument('--num_workers', default=2, type=int,
                    help='Number of workers used in data_loading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--backbone.pred_scales_num', default=3, type=int,
                    help='Number of pred scales in backbone for getting anchors')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                    help='Initial learning rate. Leave as None to read this from the config.')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum for SGD. Leave as None to read this from the config.')
parser.add_argument('--decay', '--weight_decay', default=0.0001, type=float,
                    help='Weight decay for SGD. Leave as None to read this from the config.')
parser.add_argument('--gamma', default=None, type=float,
                    help='For each lr step, what to multiply the lr by. Leave as None to read this from the config.')
parser.add_argument('--save_folder', default='weights/weights_temp/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--config', default='STMask_plus_base_config',
                    help='The config object to use.')
parser.add_argument('--save_interval', default=5000, type=int,
                    help='The number of iterations between saving the model.')
parser.add_argument('--validation_size', default=0.01, type=float,
                    help='The ratio of images to use for validation.')
parser.add_argument('--validation_epoch', default=1, type=int,
                    help='Output validation information every n iterations. If -1, do no validation.')
parser.add_argument('--output_json', dest='output_json', action='store_true',
                    help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
parser.add_argument('--keep_latest', dest='keep_latest', action='store_true',
                    help='Only keep the latest checkpoint instead of each one.')
parser.add_argument('--keep_latest_interval', default=100000, type=int,
                    help='When --keep_latest is on, don\'t delete the latest file at these intervals. This should be a multiple of save_interval or 0.')
parser.add_argument('--train_dataset', default=None, type=str,
                    help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
parser.add_argument('--no_log', dest='log', action='store_false',
                    help='Don\'t log per iteration information into log_folder.')
parser.add_argument('--log_gpu', dest='log_gpu', action='store_true',
                    help='Include GPU information in the logs. Nvidia-smi tends to be slow, so set this with caution.')
parser.add_argument('--no_interrupt', dest='interrupt', action='store_false',
                    help='Don\'t save an interrupt when KeyboardInterrupt is caught.')
parser.add_argument('--batch_alloc', default=None, type=str,
                    help='If using multiple GPUS, you can set this to be a comma separated list detailing which GPUs should get what local batch size (It should add up to your total batch size).')
parser.add_argument('--no_autoscale', dest='autoscale', action='store_false',
                    help='YOLACT will automatically scale the lr and the number of iterations depending on the batch size. Set this if you want to disable that.')

parser.set_defaults(keep_latest=False, log=True, log_gpu=False, interrupt=True, autoscale=True)
args = parser.parse_args()

if args.config is not None:
    set_cfg(args.config)

if args.train_dataset is not None:
    set_dataset(args.train_dataset, 'train')

if args.autoscale and args.batch_size*2 != 8:
    factor = args.batch_size*2 / 8
    if __name__ == '__main__':
        print('Scaling parameters by %.2f to account for a batch size of %d.' % (factor, args.batch_size))

    cfg.lr *= factor
    cfg.max_iter //= factor
    cfg.lr_steps = [x // factor for x in cfg.lr_steps]
    print('new_lr_steps:', cfg.lr_steps)


# Update training parameters from the config if necessary
def replace(name):
    if getattr(args, name) == None: setattr(args, name, getattr(cfg, name))
replace('lr')
replace('decay')
replace('gamma')
replace('momentum')
replace('backbone.pred_scales_num')

# This is managed by set_lr
cur_lr = args.lr

if torch.cuda.device_count() == 0:
    print('No GPUs detected. Exiting...')
    exit(-1)

if args.batch_size*2 // torch.cuda.device_count() < 8:
    if __name__ == '__main__':
        print('Per-GPU batch size is less than the recommended limit for batch norm. Disabling batch norm.')
    cfg.freeze_bn = True

loss_types = ['B', 'BIoU', 'C', 'M', 'T', 'center', 'B_shift', 'M_shift',
              'P', 'D', 'S', 'I']

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


class NetLoss(nn.Module):
    """
    A wrapper for running the network and computing the loss
    This is so we can more efficiently use DataParallel.
    """

    def __init__(self, net: STMask, criterion: MultiBoxLoss):
        super().__init__()

        self.net = net
        self.criterion = criterion

    def forward(self, images, gt_bboxes, gt_labels, gt_masks, gt_ids, img_meta):
        preds = self.net(images, img_meta)
        losses = self.criterion(self.net, preds, gt_bboxes, gt_labels, gt_masks, gt_ids)

        return losses


class CustomDataParallel(nn.DataParallel):
    """
    This is a custom version of DataParallel that works better with our training data.
    It should also be faster than the general case.
    """

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in device_ids] if args.cuda else None
        splits = prepare_data(inputs[0], devices, allocation=args.batch_alloc, batch_size=args.batch_size,
                              is_cuda=args.cuda, train_mode=True)

        return [[split[device_idx] for split in splits] for device_idx in range(len(devices))], \
               [kwargs] * len(devices)

    def gather(self, outputs, output_device):
        out = {}

        for k in outputs[0]:
            out[k] = torch.stack([output[k].to(output_device) for output in outputs])

        return out


def train():
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    train_dataset = get_dataset(cfg.train_dataset)

    # Parallel wraps the underlying module, but when saving and loading we don't want that
    STMask_net = STMask()
    net = STMask_net
    net.train()

    if args.log:
        log = Log(cfg.name, args.save_folder, dict(args._get_kwargs()),
                  overwrite=(args.resume is None), log_gpu_stats=args.log_gpu)

    # I don't use the timer during training (I use a different timing method).
    # Apparently there's a race condition with multiple GPUs.
    timer.disable_all()

    # Both of these can set args.resume to None, so do them before the check
    if args.resume == 'interrupt':
        args.resume = SavePath.get_interrupt(args.save_folder)
    elif args.resume == 'latest':
        args.resume = SavePath.get_latest(args.save_folder, cfg.name)

    if args.resume is not None:
        print('Resuming training, loading {}...'.format(args.resume))
        STMask_net.load_weights(path=args.resume)

        if args.start_iter == -1:
            args.start_iter = SavePath.from_str(args.resume).iteration
    else:
        print('Initializing weights based COCO ...')
        STMask_net.init_weights(backbone_path='weights/' + cfg.backbone.path)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

    criterion = MultiBoxLoss(num_classes=cfg.num_classes,
                             pos_threshold=cfg.positive_iou_threshold,
                             neg_threshold=cfg.negative_iou_threshold,
                             negpos_ratio=cfg.ohem_negpos_ratio)

    if args.batch_alloc is not None:
        args.batch_alloc = [int(x) for x in args.batch_alloc.split(',')]
        if sum(args.batch_alloc) != args.batch_size:
            print('Error: Batch allocation (%s) does not sum to batch size (%s).' % (args.batch_alloc, args.batch_size))
            exit(-1)

    net = CustomDataParallel(NetLoss(net, criterion))
    if args.cuda:
        net = net.cuda()

    # Initialize everything
    if not cfg.freeze_bn: STMask_net.freeze_bn()  # Freeze bn so we don't kill our means
    if args.cuda:
        STMask_net(torch.ones(2, 2, 3, 384, 640).cuda(), [torch.zeros(1, 4).cuda()])
    else:
        STMask_net(torch.ones(2, 2, 3, 384, 640), [torch.zeros(1, 4)])

    data_loader = data.DataLoader(train_dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True,
                                  collate_fn=detection_collate,
                                  pin_memory=True)

    # loss counters
    iteration = max(args.start_iter, 0)
    last_time = time.time()

    epoch_size = len(train_dataset) // args.batch_size
    num_epochs = math.ceil(cfg.max_iter / epoch_size)

    # Which learning rate adjustment step are we on? lr' = lr * gamma ^ step_index
    step_index = 0

    save_path = lambda epoch, iteration: SavePath(cfg.name, epoch, iteration).get_path(root=args.save_folder)
    time_avg = MovingAverage()

    global loss_types  # Forms the print order
    loss_avgs = {k: MovingAverage(100) for k in loss_types}

    print('Begin training!')
    print()
    # try-except so you can use ctrl+c to save early and stop training
    try:
        for epoch in range(num_epochs):
            # Resume from start_iter
            if (epoch + 1) * epoch_size < iteration:
                continue

            # for datum in data_loader:
            for i, data_batch in enumerate(data_loader):
                # Stop if we've reached an epoch if we're resuming from start_iter
                if iteration == (epoch + 1) * epoch_size:
                    break

                # Stop at the configured number of iterations even if mid-epoch
                if iteration == cfg.max_iter:
                    break

                # Change a config setting if we've reached the specified iteration
                changed = False
                for change in cfg.delayed_settings:
                    if iteration >= change[0]:
                        changed = True
                        cfg.replace(change[1])

                        # Reset the loss averages because things might have changed
                        for avg in loss_avgs:
                            avg.reset()

                # If a config setting was changed, remove it from the list so we don't keep checking
                if changed:
                    cfg.delayed_settings = [x for x in cfg.delayed_settings if x[0] > iteration]

                # Warm up by linearly interpolating the learning rate from some smaller value
                if cfg.lr_warmup_until > 0 and iteration <= cfg.lr_warmup_until:
                    cur_lr = (args.lr - cfg.lr_warmup_init) * (iteration / cfg.lr_warmup_until) + cfg.lr_warmup_init
                    set_lr(optimizer, cur_lr)

                # Adjust the learning rate at the given iterations, but also if we resume from past that iteration
                while step_index < len(cfg.lr_steps) and iteration >= cfg.lr_steps[step_index]:
                    step_index += 1
                    cur_lr = args.lr * (args.gamma ** step_index)
                    set_lr(optimizer, cur_lr)

                # Zero the grad to get ready to compute gradients
                optimizer.zero_grad()

                # Forward Pass + Compute loss at the same time (see CustomDataParallel and NetLoss)
                losses = net(data_batch)

                losses = {k: v.mean() for k, v in losses.items()}  # Mean here because Dataparallel
                loss = sum([losses[k] for k in losses])  # same weights in three sub-losses

                # Backprop
                loss.backward()  # Do this to free up vram even if loss is not finite
                if torch.isfinite(loss).item():
                    optimizer.step()

                # Add the loss to the moving average for bookkeeping
                for k in losses:
                    loss_avgs[k].add(losses[k].item())

                cur_time = time.time()
                elapsed = cur_time - last_time
                last_time = cur_time

                # Exclude graph setup from the timing information
                if iteration != args.start_iter:
                    time_avg.add(elapsed)

                if iteration % 10 == 0:
                    eta_str = \
                         str(datetime.timedelta(seconds=(cfg.max_iter - iteration) * time_avg.get_avg())).split('.')[0]

                    total = sum([loss_avgs[k].get_avg() for k in losses])
                    loss_labels = sum([[k, loss_avgs[k].get_avg()] for k in loss_types if k in losses], [])

                    print(('[%3d] %7d ||' + (' %s: %.3f |' * len(losses)) + ' Total: %.3f || ETA: %s || timer: %.3f')
                          % tuple([epoch, iteration] + loss_labels + [total, eta_str, elapsed]), flush=True)

                if args.log:
                    precision = 5
                    loss_info = {k: round(losses[k].item(), precision) for k in losses}
                    loss_info['Total'] = round(loss.item(), precision)

                    if args.log_gpu:
                        log.log_gpu_stats = (iteration % 10 == 0)  # nvidia-smi is sloooow

                    log.log('train', loss=loss_info, epoch=epoch, iter=iteration,
                            lr=round(cur_lr, 10), elapsed=elapsed)

                    log.log_gpu_stats = args.log_gpu

                if iteration % args.save_interval == 0 and iteration != args.start_iter:
                    if args.keep_latest:
                        latest = SavePath.get_latest(args.save_folder, cfg.name)

                    print('Saving state, iter:', iteration)
                    STMask_net.save_weights(save_path(epoch, iteration))

                    if args.keep_latest and latest is not None:
                        if args.keep_latest_interval <= 0 or iteration % args.keep_latest_interval != args.save_interval:
                            print('Deleting old save...')
                            os.remove(latest)

                # This is done per epoch
                if args.validation_epoch > 0:
                    if iteration % args.save_interval == 0 and iteration != args.start_iter:
                        setup_eval()
                        save_path_valid_metrics = save_path(epoch, iteration).replace('.pth', '.txt')
                        # valid datasets
                        metrics_valid = compute_validation_map(STMask_net, valid_data=True,
                                                               output_metrics_file=save_path_valid_metrics)
                        # valid_sub
                        # cfg.valid_sub_dataset.test_mode = False
                        # metrics = compute_validation_map(STMask_net, valid_data=False,
                        #                                  output_metrics_file=save_path_valid_metrics)

                iteration += 1

    except KeyboardInterrupt:
        print('Stopping early. Saving network...')

        # Delete previous copy of the interrupted network so we don't spam the weights folder
        SavePath.remove_interrupt(args.save_folder)

        STMask_net.save_weights(save_path(epoch, repr(iteration) + '_interrupt'))
        exit()

    STMask_net.save_weights(save_path(epoch, iteration))


def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def compute_validation_loss(net, data_loader, dataset_size):
    global loss_types
    print()
    print('compute_validation_loss, needs few minutes ...')

    with torch.no_grad():
        losses = {}
        progress_bar = ProgressBar(30, dataset_size)

        # Don't switch to eval mode because we want to get losses
        iterations, results = 0, []
        for i, data_batch in enumerate(data_loader):
            _losses = net(data_batch)

            for k, v in _losses.items():
                v = v.mean().item()
                if k in losses:
                    losses[k] += v
                else:
                    losses[k] = v

            progress = (i + 1) / dataset_size * 100
            progress_bar.set_val(i + 1)
            print('\rProcessing Images  %s %6d / %6d (%5.2f%%)'
                  % (repr(progress_bar), i + 1, dataset_size, progress), end='')

            iterations += 1

        for k in losses:
            losses[k] /= iterations

        loss_labels = sum([[k, losses[k]] for k in loss_types if k in losses], [])
        print(('Validation ||' + (' %s: %.3f |' * len(losses)) + ')') % tuple(loss_labels), flush=True)

    return loss_labels


def compute_validation_map(yolact_net, valid_data=False, output_metrics_file=None):
    with torch.no_grad():
        yolact_net.eval()
        print()
        print("Computing validation mAP (this may take a while)...", flush=True)
        metrics = eval_script.validation(yolact_net, valid_data=valid_data, output_metrics_file=output_metrics_file)
        yolact_net.train()

    return metrics


def setup_eval():
    eval_script.parse_args(['--no_bar',
                            '--batch_size=' + str(args.eval_batch_size),
                            '--output_json',
                            '--score_threshold=' + str(cfg.eval_conf_thresh),
                            '--mask_det_file='+args.save_folder+'eval_mask_det.json'])


if __name__ == '__main__':
    train()
