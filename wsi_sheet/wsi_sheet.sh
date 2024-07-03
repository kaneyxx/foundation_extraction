#!/bin/bash

# Suppose the WSIs folder structure is like this:
# --INPATH
#     |--Cancer1
#         |--*.svs
#     |--Cancer2
#         |--*.svs
# If your folder structure is different, you need to modify the code accordingly.
#
# --OUTPATH
#     |--Cancer1-PM.csv
#     |--Cancer2-PM.csv
#     |--...
#     |--Cancer1-FS.csv
#     |--Cancer2-FS.csv
#     |--...


INPATH="/n/scratch/users/f/fas994/WSI"
OUTPATH="/n/scratch/users/f/fas994/TCGA_sheets"
SLIDE="PM" # PM/FS/FS-PM

python3 wsi_sheet.py \
  --inpath ${INPATH} \
  --outpath ${OUTPATH} \
  --slide=${SLIDE}