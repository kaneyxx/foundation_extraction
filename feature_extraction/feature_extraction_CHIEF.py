###################################
#   Extract features of patches   #
###################################

import os
import torch
import torch.nn as nn
from torchvision import transforms
from models.ctran import ctranspath
import h5py
import openslide
from glob import glob
import argparse
from tqdm import tqdm
import logging

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cancer', default="01_BRCA", type=str)
    parser.add_argument('--patch_size', default=224, type=int)
    parser.add_argument('--coord_path', type=str, default=None)
    parser.add_argument('--save_root', type=str, default=None)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--log_file', type=str, default="logfile.log")
    parser.add_argument('--slide', type=str, default="PM")
    args = parser.parse_args()
    return args

def img_transform(patch_size):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    trnsfrms_val = transforms.Compose(
        [
            transforms.Resize(patch_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )
    return trnsfrms_val

if __name__ == '__main__':
    args = get_args()
    
    # Set up logging
    log_file_path = args.log_file
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file_path,  # Log to file specified by user
        filemode='w'  # Overwrite the log file each run
    )

    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    device = "cuda" if args.gpu else "cpu"
    logging.info(f"Using device: {device}")

    cancer_type = args.cancer # e.g., '04_LUAD'
    logging.info(f"Processing cancer type: {cancer_type}")
    logging.info(f"Saving output to: {args.save_root}")

    model = ctranspath()
    model.head = nn.Identity()
    td = torch.load(r'./model_weight/CHIEF_CTransPath.pth')
    model.load_state_dict(td['model'], strict=True)
    model.eval().to(device)

    # Obtain coordinates of valid patches
    h5file = glob(os.path.join(args.coord_path, f"{cancer_type}", f"{args.slide}","*.h5"))

    if not h5file:
        logging.warning(f"No h5 files found in path: output_junhan/{cancer_type}/patch_coord/")
    else:
        logging.info(f"Found {len(h5file)} h5 files to process")

    # Image settings
    patch_size = args.patch_size
    trnsfrms_val = img_transform(patch_size)

    for file in tqdm(h5file):
        file_name = file.split("/")[-1].split(".h5")[0]
        logging.info(f"Processing file: {file}")
        try:
            output_tensor = None
            with h5py.File(file, 'r') as f:
                wsi_file = f['coords'].attrs['path']
                wsi = openslide.OpenSlide(wsi_file)
                data = f['coords'][()]
                c = data.shape[0]
                logging.info(f"Extracting {c} patches from WSI file: {wsi_file}")

                for idx in range(c):
                    image = wsi.read_region(data[idx], 0, (patch_size, patch_size)).convert('RGB')
                    image = trnsfrms_val(image).unsqueeze(dim=0).to(device)
                    with torch.no_grad():
                        patch_feature_emb = model(image) # Extracted features (torch.Tensor) with shape [1,768]
                    if idx == 0:
                        output_tensor = patch_feature_emb
                    else:
                        output_tensor = torch.cat((output_tensor, patch_feature_emb), dim=0)

                # Save file
                os.makedirs(args.save_root, exist_ok=True)
                output_file_path = os.path.join(args.save_root, f"{file_name}.pt")
                torch.save(output_tensor, output_file_path)
                logging.info(f"Saved features to {output_file_path}")

        except Exception as e:
            logging.error(f"Failed to process file {file}: {str(e)}")
