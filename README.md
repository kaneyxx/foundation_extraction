# Pipeline for Extracting Features from Foundation Models

### Step 1: Create a CSV file containing the paths of whole slide images (WSIs).
Run `wsi_sheet/wsi_sheet.sh` and modify the code if needed.

### Step 2: Compute the coordinates of valid tiles.
Run `compute_coords/build_run_scripts.sh` for all sheets you created in Step 1.

* If you want, you can  easily combine Step 1 & 2.

### Step 3: Extract features using foundation models
In the **feature_extraction** folder, you can modify your own feature extraction code and the shell script to submit all extraction jobs at once.

> In Step 2 and 3, we need a text file to list all cancers(jobs) name. The intuition is that we will iterate the list to build jobs and submit them. (Refer to `tcga_all_cancer.txt file` in folder)