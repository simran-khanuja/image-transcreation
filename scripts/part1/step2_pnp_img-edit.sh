#!/bin/sh

# activate conda
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate pnp-diffusers

echo ${HOSTNAME}
export HF_HOME=/scratch/${USER}/cache

countries=("brazil" "india" "japan" "nigeria" "portugal" "turkey" "united-states")

for country in "${countries[@]}"
do
    echo "Running for $country"
    python ./src/pipelines/cap-edit/pnp-diffusers/pnp.py \
    --config_path ./configs/part1/cap-edit/$country.yaml
done