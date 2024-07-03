CANCERS=(
 TCGA-ACC   TCGA-CESC  TCGA-DLBC  TCGA-HNSC  TCGA-KIRP  
 TCGA-LUAD  TCGA-OV    TCGA-PRAD  TCGA-SKCM  TCGA-THCA  
 TCGA-UCS   TCGA-BLCA  TCGA-CHOL  TCGA-ESCA  TCGA-KICH  
 TCGA-LGG   TCGA-LUSC  TCGA-PAAD  TCGA-READ  TCGA-STAD  
 TCGA-THYM  TCGA-UVM   TCGA-BRCA  TCGA-COAD  TCGA-GBM   
 TCGA-KIRC  TCGA-LIHC  TCGA-MESO  TCGA-PCPG  TCGA-SARC  
 TCGA-TGCT  TCGA-UCEC
)
FOUNDATION="GigaPath"
SLIDE="PM"

export HF_TOKEN=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

for CANCER in "${CANCERS[@]}"; do
  python3 feature_extraction_GigaPath.py \
    --cancer ${CANCER} \
    --coord_path "/n/scratch/users/f/fas994/TCGA_coords" \
    --save_root "/n/scratch/users/f/fas994/${FOUNDATION}_features/${CANCER}-${SLIDE}" \
    --gpu \
    --log_file="/n/scratch/users/f/fas994/${FOUNDATION}_features/${CANCER}-${SLIDE}/extraction.log" \
    --slide ${SLIDE}
done