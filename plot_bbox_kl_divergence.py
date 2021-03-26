import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from layers.box_utils import gaussian_kl_divergence, jaccard, point_form
import math


def plot_gaussian(box_gt, box_pred, mask_h, mask_w, COLORS, path):
    box_gt[:, 0::2] = box_gt[:, 0::2] * mask_w
    box_gt[:, 1::2] = box_gt[:, 1::2] * mask_h
    box_pred[:, 0::2] = box_pred[:, 0::2] * mask_w
    box_pred[:, 1::2] = box_pred[:, 1::2] * mask_h

    kl_div1 = gaussian_kl_divergence(box_gt, box_pred)
    BIoU = jaccard(box_gt, box_pred).view(-1)
    plt.plot(BIoU)
    plt.plot(kl_div1)
    plt.title('blue:BIoU, orange: mean_kl')
    plt.show()
    plt.savefig(''.join([path, '_BIoU+kl']))

    box_gt = box_gt.view(-1)
    cx_gt, cy_gt = (box_gt[:2] + box_gt[2:]) / 2.0
    w_gt, h_gt = (box_gt[2:] - box_gt[:2])
    cxy_pred = (box_pred[:, :2] + box_pred[:, 2:]) / 2.0
    wh_pred = (box_pred[:, 2:] - box_pred[:, :2])

    x = np.linspace(0, mask_w, 500)
    y = np.linspace(0, mask_h, 500)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    rv = multivariate_normal([cx_gt, cy_gt], [[(w_gt/4.0)**2, 0], [0, (h_gt/4.0)**2]])
    for i in range(box_pred.size(0)):
        rv1 = multivariate_normal([cxy_pred[i][0], cxy_pred[i][1]], [[(wh_pred[i][0]/4.0)**2, 0], [0, (wh_pred[i][1]/4.0)**2]])

        fig1 = plt.figure()
        ax = fig1.gca(projection='3d')
        ax.plot_surface(X, Y, rv.pdf(pos), cmap='RdBu_r', linewidth=0)
        ax.plot_surface(X, Y, rv1.pdf(pos), cmap='RdBu_r', linewidth=0)
        plt.title([i, ('BIoU:', BIoU[i], 'KL', kl_div[i].tolist())])
        plt.savefig(''.join([path, str(i), '_gaussian']))
        # plt.show()

        fig3 = plt.figure()
        ax1 = fig3.add_subplot(111)
        ax1.contour(X, Y, rv.pdf(pos), colors='k', linewidths=0.5)
        ax1.contour(X, Y, rv1.pdf(pos))

        x = np.full((mask_h, mask_w, 3), [255, 255, 255], np.uint8)
        cv2.rectangle(x, (box_gt[0], box_gt[1]), (box_gt[2], box_gt[3]), [0, 0, 0], 1)  # black
        cv2.rectangle(x, (box_pred[i][0], box_pred[i][1]), (box_pred[i][2], box_pred[i][3]), COLORS[i + 1], 1)
        ax1.imshow(x)
        plt.title([i, ('BIoU:', BIoU[i], 'KL', kl_div[i].tolist())])
        plt.savefig(''.join([path, str(i), '_bbox']))
        # plt.show()
        # plt.savefig(''.join([path, str(i), '_gaussian_contour']))
        # plt.show()


def plot_kl_div(path):
    mu_gt, sigma_gt = 0, 1
    # x = torch.linspace(-1, 1, 20)
    x = torch.zeros(1).repeat(20)
    y = torch.linspace(0.6, 4, 20)
    mu_pred, sigma_pred = torch.meshgrid(x, y)

    kl_div0 = (sigma_pred / sigma_gt) ** 2 + (mu_pred - mu_gt) ** 2 / sigma_gt ** 2 - 1 \
              - 2 * torch.log(sigma_pred / sigma_gt)

    kl_div1 = (sigma_gt / sigma_pred) ** 2 + (mu_gt - mu_pred) ** 2 / sigma_pred ** 2 - 1 \
              - 2 * torch.log(sigma_gt / sigma_pred)
    # kl_div = 0.5 * (kl_div0 + kl_div1) / 2

    CE1 = (sigma_gt / sigma_pred) ** 2 + (mu_pred - mu_gt) ** 2 / sigma_pred ** 2 \
        + 2 * torch.log((2 * math.pi) / sigma_pred)

    kl_div = 0.5 * kl_div1

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_wireframe(mu_pred, sigma_pred, kl_div)
    ax.view_init(15, 0)
    plt.title(''.join(['gt: mu=', str(mu_gt), ', sigma=', str(sigma_gt)]))
    # plt.savefig(''.join([path, '2D_kl_divergence_', str(mu_gt), str(sigma_gt), '_D(gt | pred).png']))
    plt.show()


COLORS = ((244, 67, 54),
          (233, 30, 99),
          (156, 39, 176),
          (103, 58, 183),
          (63, 81, 181),
          (33, 150, 243),
          (3, 169, 244),
          (0, 188, 212),
          (0, 150, 136),
          (76, 175, 80),
          (139, 195, 74),
          (205, 220, 57),
          (255, 235, 59),
          (255, 193, 7),
          (255, 152, 0),
          (255, 87, 34),
          (121, 85, 72),
          (158, 158, 158),
          (96, 125, 139))
mask_h, mask_w = 96, 160
# [cx, cy, w, h]
box_gt = torch.tensor([0.5, 0.5, 0.4, 0.4]).view(1, -1)
n = 11
w = torch.cat([torch.linspace(0.5, 1, 6)[:-1], torch.linspace(1, 2, 6)]).view(-1, 1)
box_pred = box_gt.repeat(n, 1)
box_pred[:, 2:] = box_pred[:, 2:] * w
box_pred = torch.clamp(box_pred, min=0, max=1)
path = '/home/lmh/Downloads/VIS/code/yolact_JDT_VIS/results/kl_div/'

box_gt = point_form(box_gt)
box_pred = point_form(box_pred)
plot_gaussian(box_gt, box_pred, mask_h, mask_w, COLORS, path)
# plot_kl_div(path)
