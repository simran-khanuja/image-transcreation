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
    clip-retrieval index \
    --embeddings_folder ${LAION_ROOT}/embeddings/ViT-H-14/${country}_en \
    --index_folder ${LAION_ROOT}/index/ViT-H-14/${country}_en
done