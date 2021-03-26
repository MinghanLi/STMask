import numpy as np
import os.path as osp
import random
import mmcv
from .custom import CustomDataset
from .extra_aug import ExtraAugmentation
from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         Numpy2Tensor)
from cocoapi.PythonAPI.pycocotools.ytvos import YTVOS
from mmcv.parallel import DataContainer as DC
from .utils import to_tensor, random_scale
import torch


class YTVOSDataset(CustomDataset):
    CLASSES = ('person', 'giant_panda', 'lizard', 'parrot', 'skateboard', 'sedan',
                'ape', 'dog', 'snake', 'monkey', 'hand', 'rabbit', 'duck', 'cat', 'cow', 'fish',
                'train', 'horse', 'turtle', 'bear', 'motorbike', 'giraffe', 'leopard',
                'fox', 'deer', 'owl', 'surfboard', 'airplane', 'truck', 'zebra', 'tiger',
                'elephant', 'snowboard', 'boat', 'shark', 'mouse', 'frog', 'eagle', 'earless_seal',
                'tennis_racket')

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=True,
                 with_crowd=False,
                 with_label=True,
                 with_track=False,
                 extra_aug=None,
                 aug_ref_bbox_param=None,
                 resize_keep_ratio=True,
                 test_mode=False,
                 clip_frames=1):
        # prefix of images path
        self.img_prefix = img_prefix

        # load annotations (and proposals)
        self.vid_infos = self.load_annotations(ann_file)
        img_ids = []
        for idx, vid_info in enumerate(self.vid_infos):
          for frame_id in range(len(vid_info['filenames'])):
            img_ids.append((idx, frame_id))
        self.img_ids = img_ids
        if proposal_file is not None:
            self.proposals = self.load_proposals(proposal_file)
        else:
            self.proposals = None
        # filter images with no annotation during training
        if not test_mode:
            valid_inds = [i for i, (v, f) in enumerate(self.img_ids)
                if len(self.get_ann_info(v, f)['bboxes'])]
            self.img_ids = [self.img_ids[i] for i in valid_inds]

        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        # normalization configs
        self.img_norm_cfg = img_norm_cfg
        self.clip_frames = clip_frames

        # max proposals per image
        self.num_max_proposals = num_max_proposals
        # flip ratio
        self.flip_ratio = flip_ratio
        assert flip_ratio >= 0 and flip_ratio <= 1
        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor

        # with mask or not (reserved field, takes no effect)
        self.with_mask = with_mask
        # some datasets provide bbox annotations as ignore/crowd/difficult,
        # if `with_crowd` is True, then these info is returned.
        self.with_crowd = with_crowd
        # with label is False for RPN
        self.with_label = with_label
        self.with_track = with_track
        # params for augmenting bbox in the reference frame
        self.aug_ref_bbox_param = aug_ref_bbox_param
        # in test mode or not
        self.test_mode = test_mode

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # transforms
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.bbox_transform = BboxTransform()
        self.mask_transform = MaskTransform()
        self.numpy2tensor = Numpy2Tensor()

        # if use extra augmentation
        if extra_aug is not None:
            self.extra_aug = ExtraAugmentation(**extra_aug)
        else:
            self.extra_aug = None

        # image rescale if keep ratio
        self.resize_keep_ratio = resize_keep_ratio

    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(self.img_ids[idx])
        data = self.prepare_train_img(self.img_ids[idx])
        return data
    
    def load_annotations(self, ann_file):
        self.ytvos = YTVOS(ann_file)
        self.cat_ids = self.ytvos.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.vid_ids = self.ytvos.getVidIds()
        vid_infos = []
        for i in self.vid_ids:
            info = self.ytvos.loadVids([i])[0]
            info['filenames'] = info['file_names']
            vid_infos.append(info)
        return vid_infos

    def get_ann_info(self, idx, frame_id):
        vid_id = self.vid_infos[idx]['id']
        ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
        ann_info = self.ytvos.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info, frame_id)

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            vid_id, _ = self.img_ids[i]
            vid_info = self.vid_infos[vid_id]
            if vid_info['width'] / vid_info['height'] > 1:
                self.flag[i] = 1

    def bbox_aug(self, bbox, img_size):
        assert self.aug_ref_bbox_param is not None
        center_off = self.aug_ref_bbox_param[0]
        size_perturb = self.aug_ref_bbox_param[1]
        
        n_bb = bbox.shape[0]
        # bbox center offset
        center_offs = (2*np.random.rand(n_bb, 2) - 1) * center_off
        # bbox resize ratios
        resize_ratios = (2*np.random.rand(n_bb, 2) - 1) * size_perturb + 1
        # bbox: x1, y1, x2, y2
        centers = (bbox[:,:2]+ bbox[:,2:])/2.
        sizes = bbox[:,2:] - bbox[:,:2]
        new_centers = centers + center_offs * sizes
        new_sizes = sizes * resize_ratios
        new_x1y1 = new_centers - new_sizes/2.
        new_x2y2 = new_centers + new_sizes/2.
        c_min = [0,0]
        c_max = [img_size[1], img_size[0]]
        new_x1y1 = np.clip(new_x1y1, c_min, c_max)
        new_x2y2 = np.clip(new_x2y2, c_min, c_max)
        bbox = np.hstack((new_x1y1, new_x2y2)).astype(np.float32)
        return bbox

    def sample_ref(self, idx):
        # sample another frame in the same sequence as reference
        vid, frame_id = idx
        vid_info = self.vid_infos[vid]
        sample_range = range(len(vid_info['filenames']))
        valid_samples = []
        for i in range(-2*self.clip_frames, 2*self.clip_frames+1):
            # check if the frame id is valid
            ref_idx = (vid, i+frame_id)
            if i != 0 and ref_idx in self.img_ids:
                valid_samples.append(i+frame_id)
        if len(valid_samples) == 0:
            ref_frames = [frame_id]
        else:
            ref_frames = random.sample(valid_samples, 1)
        return ref_frames

    def prepare_train_img(self, idx):
        # prepare a pair of image in a sequence
        vid,  frame_id = idx
        vid_info = self.vid_infos[vid]
        basename = osp.basename(vid_info['filenames'][frame_id])
        clip_frame_ids = self.sample_ref(idx) + [frame_id]
        clip_frame_ids.sort()
        imgs = []
        for frame_id in clip_frame_ids:
            imgs.append(mmcv.imread(osp.join(self.img_prefix, vid_info['filenames'][frame_id])))
        imgs = np.stack(imgs, axis=0)
        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        # load annotation of ref_frames
        bboxes, labels, ids, masks, bboxes_ignore = [], [], [], [], []
        for frame_id in clip_frame_ids:
            ann = self.get_ann_info(vid, frame_id)
            bboxes.append(ann['bboxes'])
            labels.append(ann['labels'])
            # obj ids attribute does not exist in current annotation
            # need to add it
            ids.append(ann['obj_ids'])
            if self.with_mask:
                masks.append(ann['masks'])
            # compute matching of reference frame with current frame
            # 0 denote there is no matching
            # gt_pids = [ref_ids.index(i)+1 if i in ref_ids else 0 for i in gt_ids]
            if self.with_crowd:
                bboxes_ignore.append(ann['bboxes_ignore'])

        # extra augmentation
        if self.extra_aug is not None and self.with_mask:
            for i in range(len(clip_frame_ids)):
                imgs[i], bboxes[i], labels[i], masks[i], ids[i] = self.extra_aug(imgs[i], bboxes[i], labels[i], masks[i], ids[i])

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        img_scale = random_scale(self.img_scales, mode='range_keep_ratio')  # sample a scale
        temp_imgs = []
        for i in range(len(clip_frame_ids)):
            img_cur, img_shape, pad_shape, scale_factor = self.img_transform(
                imgs[i], img_scale, flip, keep_ratio=self.resize_keep_ratio)
            temp_imgs.append(img_cur)
        imgs = np.stack(temp_imgs, axis=0)
        imgs = imgs.copy()
        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape, scale_factor, flip)
            proposals = np.hstack([proposals, scores]) if scores is not None else proposals

        for i in range(len(clip_frame_ids)):
            bboxes[i] = self.bbox_transform(bboxes[i], img_shape, pad_shape, scale_factor, flip)
        if self.aug_ref_bbox_param is not None:
            for i in range(len(clip_frame_ids)):
                bboxes[i] = self.bbox_aug(bboxes[i], img_shape)
        if self.with_crowd:
            for i in range(len(clip_frame_ids)):
                bboxes_ignore[i] = self.bbox_transform(bboxes_ignore[i], img_shape, pad_shape, scale_factor, flip)
        if self.with_mask:
            for i in range(len(clip_frame_ids)):
                masks[i] = self.mask_transform(masks[i], pad_shape, img_scale, flip, keep_ratio=self.resize_keep_ratio)

        ori_shape = (vid_info['height'], vid_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            video_id=vid,
            frame_id=frame_id,
            is_first=(frame_id == 0),
            scale_factor=scale_factor,
            flip=flip)

        data = dict(
            img=DC(to_tensor(imgs), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            bboxes=DC([to_tensor(bboxes[i]) for i in range(len(clip_frame_ids))]),
        )
        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['labels'] = DC([to_tensor(labels[i]) for i in range(len(clip_frame_ids))])
        if self.with_track:
            data['ids'] = DC([to_tensor(np.array(ids[i])) for i in range(len(clip_frame_ids))])
        if self.with_crowd:
            data['bboxes_ignore'] = DC([to_tensor(bboxes_ignore[i]) for i in range(len(clip_frame_ids))])
        if self.with_mask:
            data['masks'] = DC([to_tensor(masks[i]) for i in range(len(clip_frame_ids))], cpu_only=True)
        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        vid, frame_id = idx
        vid_info = self.vid_infos[vid]
        img = mmcv.imread(osp.join(self.img_prefix, vid_info['filenames'][frame_id]))
        proposal = None

        def prepare_single(img, frame_id, scale, flip, proposal=None):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(vid_info['height'], vid_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                is_first=(frame_id == 0),
                video_id=vid,
                frame_id=frame_id,
                scale_factor=scale_factor,
                flip=flip)
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack(
                    [_proposal, score]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            _img, _img_meta, _proposal = prepare_single(
                img, frame_id, scale, False, proposal)
            imgs.append(DC(_img))
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            # if self.flip_ratio > 0:
            #     _img, _img_meta, _proposal = prepare_single(
            #         img, frame_id, scale, True, proposal)
            #     imgs.append(DC(_img))
            #     img_metas.append(DC(_img_meta, cpu_only=True))
            #     proposals.append(_proposal)
        data = dict(img=imgs, img_meta=img_metas)
        return data

    def sample_ref_test(self, idx):
        # sample another frame in the same sequence as reference
        vid, frame_id = idx
        vid_info = self.vid_infos[vid]
        sample_range = range(len(vid_info['filenames']))
        valid_samples = []
        if frame_id == 0:
            for i in sample_range:
                if len(valid_samples) >= self.eval_clip_frames:
                    break
                # check if the frame id is valid
                ref_idx = (vid, i+frame_id)
                if ref_idx in self.img_ids:
                    valid_samples.append(i+frame_id+1)
        else:
            ref_idx = (vid, frame_id+self.eval_clip_frames)
            if ref_idx in self.img_ids:
                valid_samples.append(frame_id+self.eval_clip_frames)
        return valid_samples

    def _parse_ann_info(self, ann_info, frame_id, with_mask=True):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        gt_bboxes = []
        gt_labels = []
        gt_ids = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []
        for i, ann in enumerate(ann_info):
            # each ann is a list of masks
            # ann:
            # bbox: list of bboxes
            # segmentation: list of segmentation
            # category_id
            # area: list of area
            bbox = ann['bboxes'][frame_id]
            area = ann['areas'][frame_id]
            segm = ann['segmentations'][frame_id]
            if bbox is None: continue
            x1, y1, w, h = bbox
            if area <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_ids.append(ann['id'])
                gt_labels.append(self.cat2label[ann['category_id']])
            if with_mask:
                gt_masks.append(self.ytvos.annToMask(ann, frame_id))
                mask_polys = [
                    p for p in segm if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                poly_lens = [len(p) for p in mask_polys]
                gt_mask_polys.append(mask_polys)
                gt_poly_lens.extend(poly_lens)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, obj_ids=gt_ids, bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens
        return ann
