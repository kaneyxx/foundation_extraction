import glob
import os
import pandas as pd
from argparse import ArgumentParser
import re


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--inpath",
        help="""Input Path """,
        type=str,
        default="./WSI"
    )
    parser.add_argument(
        "--outpath",
        help="""Output Path """,
        type=str,
        default="./data_sheet"
        )
    parser.add_argument(
        "--slide",
        help="""Slide type: FS, PM, FS-PM""",
        type=str,
        default="PM"
        )
    args = parser.parse_args()

    base = args.inpath
    outpath = args.outpath
    # You need to modify the folder pattern to search those WSIs.
    cancer_folders = glob.glob(os.path.join(base, "TCGA-*"))
    count = 0
    for folder in cancer_folders:
        cancer_type = folder.split("/")[-1]
        os.makedirs(os.path.join(outpath, args.slide), exist_ok=True)
        WSIs = glob.glob(os.path.join(base, cancer_type, '*.svs'))

        pattern = r'-DX[0-9A-Z]+\.'
        if args.slide=="PM":
            selected_WSIs = [WSI for WSI in WSIs if re.search(pattern, WSI)]
            count += len(selected_WSIs)
        elif args.slide=="FS":
            selected_WSIs = [WSI for WSI in WSIs if not re.search(pattern, WSI)]
            count += len(selected_WSIs)
        else:
            selected_WSIs = WSIs
            count += len(selected_WSIs)
        outfile_name = f"{cancer_type}-{args.slide}.csv"

        df = pd.DataFrame({'WSI_path': selected_WSIs})
        outfile = os.path.join(outpath, outfile_name)
        df.to_csv(outfile, index=False)
    print(f"processed:{count} WSIs")