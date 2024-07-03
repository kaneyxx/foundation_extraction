# -*- coding: utf-8 -*-
# **********************************
# Author: scusenyang              #
# Email:  scusenyang@tencent.com  #
# **********************************
import os
import argparse
import math
import glob
import logging
import csv
import pandas as pd
from itertools import islice
from multiprocessing.pool import ThreadPool
import cv2
import h5py
import openslide
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.color import rgb2hsv
import SimpleITK

from tqdm import tqdm

def compute_patch_size(wsi, target_mpp, target_patch_size, downsample_rate,mpp=None):

    # compute mpp of wsi

    if mpp is None:
        if wsi.properties.get("openslide.mpp-x"):
            mpp = float(wsi.properties.get("openslide.mpp-x", 1))
            print("#1 mpp:", mpp)
        else:
            unit = wsi.properties.get("tiff.ResolutionUnit")
            x_resolution = float(wsi.properties.get("tiff.XResolution"))
            if unit.lower() == "centimeter":
                mpp = 10000 / x_resolution
            else:
                mpp = 25400 / x_resolution
            print("#2 mpp:", mpp)   #  panda 0.5031  0.45  0.48

        # mpp = 0.25   #
            # print("# /mnt/group-ai-medical-2/private/scusenyang/data_tencent/tissuenet/train/tif_images")
        # mpp = float(csv_mpp)  #

        print("# mpp = ", mpp)
        if mpp > 1:
            level0_size = int(target_mpp / float(mpp) * 2 * target_patch_size / 2)
        else:
            level0_size = int(round(target_mpp / float(mpp)) / 2 * target_patch_size * 2)

        print("level0_size", level0_size)

    level0_size = round(target_mpp / float(mpp) / 2) * target_patch_size * 2
    print(level0_size)
    # level0_size = round(target_mpp / float(mpp) / 2) * args.patch_size *2

    return level0_size // downsample_rate

def save_coords_h5(coords, patch_size, h5_path):
    file = h5py.File(h5_path, 'w')
    dset = file.create_dataset('coords',data=coords)
    dset.attrs['patch_size'] = patch_size

    file.close()

def save_coords_h5_modified(coords, patch_size, h5_path, original_path):
    file = h5py.File(h5_path, 'w')
    dset = file.create_dataset('coords',data=coords)
    dset.attrs['patch_size'] = patch_size
    dset.attrs['path'] = original_path
    file.close()

def get_thumbnail(wsi, downsample=16):
    full_size = wsi.dimensions
    img_rgb = np.array(wsi.get_thumbnail((int(full_size[0] / downsample), int(full_size[1] / downsample))))
    return img_rgb

def get_thumbnail_png(wsi, downsample=16):
    width = int(wsi.shape[1] // downsample)
    height = int(wsi.shape[0] // downsample)

    dim = (width, height)

    img_rgb = cv2.resize(wsi, dim, interpolation=cv2.INTER_AREA)
    # full_size = wsi.dimensions
    # img_rgb = np.array(wsi.get_thumbnail((int(full_size[0] / downsample), int(full_size[1] / downsample))))
    return img_rgb

def get_tissue_mask(img_RGB):
    """

    :param img_RGB: numpy array of RGB image
    :return:
    """
    # remove black background
    img_RGB[np.all(img_RGB <= (50, 50, 50), axis=-1)] = (255,255,255)

    # img = np.array(self.wsi.read_region((0, 0), seg_level, self.level_dim[seg_level]))

    img=img_RGB
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space
    img_med = cv2.medianBlur(img_hsv[:, :, 1], 11)  # Apply median blurring

    _, img_otsu = cv2.threshold(img_med, 15, 255, cv2.THRESH_BINARY)


    tissue_mask = img_otsu.astype(np.uint8)
    return tissue_mask, img_otsu, img_otsu, img_otsu, img_otsu, img_otsu


def extract_useful_patches(tissue_mask, patch_size, threshold):

    useful_patches_coord = []


    for x in np.arange(0, tissue_mask.shape[1], patch_size):
        for y in np.arange(0,  tissue_mask.shape[0], patch_size):
            # last patch shall reach just up to the last pixel
            if (x + patch_size > tissue_mask.shape[1]):
                x = tissue_mask.shape[1]- patch_size

            if (y + patch_size > tissue_mask.shape[0]):
                y = tissue_mask.shape[0] - patch_size

            patch = tissue_mask[y: y + patch_size, x: x + patch_size]
            if patch.mean() > threshold:
                useful_patches_coord.append([x, y])

    return useful_patches_coord


def compute_coords_single(wsi_path, patch_coord_dir, visual_mask_dir, visual_stitch_dir, args):
    # slide_id = '.'.join(os.path.basename(wsi_path[0]).split('.')[:-1])
    slide_id = '.'.join(os.path.basename(wsi_path).split('.')[:-1])
    patch_coord_h5 = os.path.join(patch_coord_dir, slide_id + '.h5')
    visual_mask = os.path.join(visual_mask_dir, slide_id + '.jpg')
    visual_stitch = os.path.join(visual_stitch_dir, slide_id + '.jpg')

    if args.set_png:

        wsi_image = SimpleITK.ReadImage(wsi_path)

        wsi = SimpleITK.GetArrayFromImage(wsi_image)
        img_rgb = get_thumbnail_png(wsi, downsample=args.downsample)
        tissue_mask, tissue_S, tissue_RGB, min_R, min_G, min_B = get_tissue_mask(img_rgb) # x -> d1, y -> d2
        print(tissue_mask.shape)
    else:


    # wsi = openslide.open_slide(wsi_path[0])
        wsi = openslide.open_slide(wsi_path)
        img_rgb = get_thumbnail(wsi, downsample=args.downsample)

        tissue_mask, tissue_S, tissue_RGB, min_R, min_G, min_B = get_tissue_mask(img_rgb) # x -> d1, y -> d2

    if args.set_mpp:

        patch_size_downsample = compute_patch_size(wsi, args.target_mpp, args.patch_size, args.downsample,args.mpp)

    else:
        patch_size_downsample = compute_patch_size(wsi, args.target_mpp, args.patch_size, args.downsample)

    coords_downsample = extract_useful_patches(
        tissue_mask, patch_size_downsample, args.threshold
    )


    coords = np.array(coords_downsample) * args.downsample

    # save_coords_h5(coords, patch_size_downsample * args.downsample, patch_coord_h5)
    save_coords_h5_modified(coords, patch_size_downsample * args.downsample, patch_coord_h5, wsi_path)

def get_result_dirs(result_root):
    patch_coord_dir = os.path.join(result_root, 'patch_coord')
    patch_feature_dir = os.path.join(result_root, 'patch_feature')
    visual_mask_dir = os.path.join(result_root, 'visual_mask')
    visual_stitch_dir = os.path.join(result_root, 'visual_stitch')

    if not os.path.isdir(patch_coord_dir):
        os.makedirs(patch_coord_dir)
    if not os.path.isdir(patch_feature_dir):
        os.makedirs(patch_feature_dir)
    if not os.path.isdir(visual_mask_dir):
        os.makedirs(visual_mask_dir)
    if not os.path.isdir(visual_stitch_dir):
        os.makedirs(visual_stitch_dir)

    return patch_coord_dir, patch_feature_dir, visual_mask_dir, visual_stitch_dir

def get_wsi_path(args):
    if args.csv_path is not None:

        df = pd.read_csv(args.csv_path)
        wsi_path_list = list(df['WSI_path'].values)
    else:
        wsi_path_list = sorted(glob.glob(args.wsi_path))

    fold_size = math.ceil(len(wsi_path_list) / args.n_part)
    wsi_path_list = wsi_path_list[fold_size * args.part: fold_size * args.part + fold_size]
    # mpp_list = mpp_list[fold_size * args.part: fold_size * args.part + fold_size]   #
    
    
    return wsi_path_list

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--target_mpp', default=1.0, type=float)
    parser.add_argument('--patch_size', default=224, type=int)
    parser.add_argument('--downsample', default=16, type=int)
    parser.add_argument('--threshold', default=0.15, type=float)
    parser.add_argument('--num_worker', default=20, type=int)
    parser.add_argument('--save_root', type=str, default=None)
    parser.add_argument('--wsi_path', type=str, default=None)
    parser.add_argument('--wsi_type', type=str, default=None)
    parser.add_argument('--csv_path', type=str, default=None)
    parser.add_argument('--n_part', type=int, default=1)
    parser.add_argument('--part', type=int, default=0)
    parser.add_argument('--column', type=str, default='slide_path')
    parser.add_argument('--mpp', type=float, default=0.5)
    parser.add_argument('--set_mpp', action='store_true')
    parser.add_argument('--set_png', action='store_true')
    args = parser.parse_args()

    return args


def multi_thread_process(wsi_path, patch_coord_dir, visual_mask_dir, visual_stitch_dir, pbar, logger, args):
    # slide_id = '.'.join(os.path.basename(wsi_path[0]).split('.')[:-1])  #
    slide_id = '.'.join(os.path.basename(wsi_path).split('.')[:-1])  #
    
    if os.path.exists(os.path.join(patch_coord_dir, f"{slide_id}.h5")):
        # logger.info(f"{slide_id} exist.")
        pbar.update(1)
        return
    else:
        try:
            logger.info(f"Processing: {slide_id}")
            compute_coords_single(wsi_path, patch_coord_dir, visual_mask_dir, visual_stitch_dir, args)
            logger.info(f"Complete!")
        except Exception as e:
            logger.error(e)
            logger.info(f"Invalid WSI: {slide_id}")
        pbar.update(1)
def get_logger(name, save_path):

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(save_path)
    fh.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)

    format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(format)
    sh.setFormatter(format)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger

if __name__ == '__main__':

    args = get_args()

    # create storage space
    patch_coord_dir, patch_feature_dir, visual_mask_dir, visual_stitch_dir = get_result_dirs(args.save_root)

    logger = get_logger('', os.path.join(args.save_root, 'seg_log.txt'))

    # get wsi list
    wsi_path_list= get_wsi_path(args)
    print(wsi_path_list)

    pbar = tqdm(total=len(wsi_path_list))
    pool = ThreadPool(args.num_worker)
    iterable = [[wsi_path, patch_coord_dir, visual_mask_dir, visual_stitch_dir, pbar, logger, args] for wsi_path in wsi_path_list]
    pool.starmap(multi_thread_process, iterable)
    pool.close()
