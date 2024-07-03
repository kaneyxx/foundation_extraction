###################################
#   Extract features of patches   #
###################################

import os
import torch
import torch.nn as nn
from torchvision import transforms
import timm
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
    transform = transforms.Compose(
    [
        transforms.Resize(patch_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    )
    return transform

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
    
    ### -------------------------------------------------------###
    # Modify this part to your model and transformation function #
    ### -------------------------------------------------------###
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    model = model.eval().to(device)
    transform = img_transform(args.patch_size)

    # Obtain coordinates of valid patches
    h5file = glob(os.path.join(args.coord_path, f"{cancer_type}-{args.slide}", "patch_coord", "*.h5"))

    if not h5file:
        logging.warning(f"No h5 files found in path: {args.coord_path}")
    else:
        logging.info(f"Found {len(h5file)} h5 files to process")

    # Image settings
    patch_size = args.patch_size

    for file in tqdm(h5file):
        file_name = file.split("/")[-1].split(".h5")[0]
        output_file_path = os.path.join(args.save_root, f"{file_name}.pt")

        # Check if the output file already exists
        if os.path.exists(output_file_path):
            logging.info(f"File {output_file_path} already exists. Skipping.")
            continue
        
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
                    image = transform(image).unsqueeze(dim=0).to(device)
                    with torch.no_grad():
                        patch_feature_emb = model(image) # Extracted features (torch.Tensor) with shape [1, dim]
                    if idx == 0:
                        output_tensor = patch_feature_emb
                    else:
                        output_tensor = torch.cat((output_tensor, patch_feature_emb), dim=0)

                # Save file
                os.makedirs(args.save_root, exist_ok=True)
                torch.save(output_tensor, output_file_path)
                logging.info(f"Saved features to {output_file_path}")

        except Exception as e:
            logging.error(f"Failed to process file {file}: {str(e)}")
