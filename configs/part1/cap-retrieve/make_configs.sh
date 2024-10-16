#!/bin/bash

# Define the countries
declare -A country_caps
country_caps["india"]="India"
country_caps["japan"]="Japan"
country_caps["portugal"]="Portugal"
country_caps["brazil"]="Brazil"
country_caps["nigeria"]="Nigeria"
country_caps["united-states"]="United-States"
country_caps["turkey"]="Turkey"

# Template for the config file
config_template="indice_folder: ./datacomp/index/ViT-H-14/COUNTRY_CAP_en
metadata_path: ./outputs/part1/caption-llm_edit/COUNTRY/metadata.csv
clip_model: open_clip:ViT-H-14
indices_path: ./datacomp/index/ViT-H-14/indices_paths.json
indice_name: COUNTRY_en
src_image_path_col: src_image_path
caption_col: caption
llm_edit_col: llm_edit
src_country_col: src_country
jsonl_file_path: ./datacomp/jsonl/laion2B-en/COUNTRY_CAP_dataset.jsonl
output_file: ./outputs/part1/cap-retrieve/COUNTRY/metadata.csv
tgt_image_path: ./outputs/part1/cap-retrieve/COUNTRY"

# Iterate over the countries and create config files
for country in "${!country_caps[@]}"
do
    # Get the capitalized version of the country name
    country_cap=${country_caps[$country]}

    # Replace 'COUNTRY' and 'COUNTRY_CAP' with the actual country name in the template
    country_config="${config_template//COUNTRY_CAP/$country_cap}"
    country_config="${country_config//COUNTRY/$country}"
    

    # Write the config to a file
    echo "$country_config" > "./configs/part1/cap-retrieve/${country}.yaml"
done

echo "Config files created for all countries."
