#!/bin/sh

ROOT="./datacomp"

# activate conda
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate transcreation

echo ${HOSTNAME}
mkdir -p /scratch/${USER}/cache

export HF_HOME=/scratch/${USER}/cache

# Predefined list of countries
countries=("brazil" "india" "japan" "nigeria" "portugal" "turkey" "united-states")

# Read the input argument (comma-separated list of countries)
input_countries="$1"

# Check if the input is 'all', if so, use the predefined list of countries
if [[ "$input_countries" == "all" ]]; then
    selected_countries=("${countries[@]}")
else
    # Convert the comma-separated input into an array
    IFS=',' read -r -a selected_countries <<< "$input_countries"
fi

# Iterate over the selected countries and run the Python script
for country in "${selected_countries[@]}"
do
    clip-retrieval index \
    --embeddings_folder ${ROOT}/embeddings/ViT-H-14/${country}_en \
    --index_folder ${ROOT}/index/ViT-H-14/${country}_en
done