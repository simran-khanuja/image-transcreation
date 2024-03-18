#!/bin/sh

LAION_ROOT="./laion"

# activate conda
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate image-transcreation

echo ${HOSTNAME}

mkdir -p /scratch/${USER}/cache
export HF_HOME=/scratch/${USER}/cache

countries=("brazil" "india" "japan" "nigeria" "portugal" "turkey" "united-states")

for country in "${countries[@]}"
do 
    # Get brace pattern for all tar files in the folder
    # Extract numerical parts from filenames and sort them
    img2dataset_path="${LAION_ROOT}/img2dataset/${country}_en"
    numbers=$(ls ${img2dataset_path}/*.tar | xargs -n 1 basename | sed 's/.tar//g' | sort -n)

    # Get the minimum and maximum values
    min=$(echo "$numbers" | head -n 1)
    max=$(echo "$numbers" | tail -n 1)

    # Construct the brace pattern
    brace_pattern="{${min}..${max}}.tar"

    clip-retrieval inference \
    --input_dataset="${LAION_ROOT}/img2dataset/${country}_en/${brace_pattern}"  \
    --output_folder="${LAION_ROOT}/embeddings/ViT-H-14/${country}_en" \
    --input_format="webdataset" \
    --cache_path="/scratch/${USER}/cache" \
    --clip_model="open_clip:ViT-H-14" \
    --clip_cache_path="/scratch/${USER}/cache" \
    --enable_metadata=True
done