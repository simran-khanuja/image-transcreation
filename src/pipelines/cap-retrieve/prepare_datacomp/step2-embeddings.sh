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
    # Get brace pattern for all tar files in the folder
    # Extract numerical parts from filenames and sort them
    img2dataset_path="${ROOT}/img2dataset/${country}_en"
    numbers=$(ls ${img2dataset_path}/*.tar | xargs -n 1 basename | sed 's/.tar//g' | sort -n)

    # Get the minimum and maximum values
    min=$(echo "$numbers" | head -n 1)
    max=$(echo "$numbers" | tail -n 1)

    # Construct the brace pattern
    brace_pattern="{${min}..${max}}.tar"

    clip-retrieval inference \
    --input_dataset="${ROOT}/img2dataset/${country}_en/${brace_pattern}"  \
    --output_folder="${ROOT}/embeddings/ViT-H-14/${country}_en" \
    --input_format="webdataset" \
    --cache_path="/scratch/${USER}/cache" \
    --clip_model="open_clip:ViT-H-14" \
    --clip_cache_path="/scratch/${USER}/cache" \
    --enable_metadata=True
done