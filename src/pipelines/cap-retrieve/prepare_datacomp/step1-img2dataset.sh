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
    img2dataset --url_list=${ROOT}/jsonl/datacomp/${country}_dataset.jsonl \
    --output_folder=${ROOT}/img2dataset/${country}_en \
    --processes_count=8 --image_size=256 --input_format jsonl \
    --output_format webdataset --url_col URL --caption_col TEXT --thread_count 32
done