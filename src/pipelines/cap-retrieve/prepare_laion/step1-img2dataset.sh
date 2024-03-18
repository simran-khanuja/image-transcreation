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
    img2dataset --url_list=${LAION_ROOT}/jsonl/laion2B-en/${country}_dataset.jsonl \
    --output_folder=${LAION_ROOT}/img2dataset/${country}_en \
    --processes_count=8 --image_size=256 --input_format jsonl \
    --output_format webdataset --url_col URL --caption_col TEXT --thread_count 32
done