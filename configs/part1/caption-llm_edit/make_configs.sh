#!/bin/bash

OPENAI_API_KEY=""
ROOT_DIR="/home/${USER}/image-transcreation"

# Define the countries
countries=("brazil" "india" "japan" "nigeria" "portugal" "turkey" "united-states")
country_names=("Brazil" "India" "Japan"  "Nigeria" "Portugal" "Turkey" "the United States")

# Template for the config file
config_template="source_countries: SOURCE_COUNTRIES
source_data_path: $ROOT_DIR/data/part1
output_dir: $ROOT_DIR/outputs/part1/caption-llm_edit/COUNTRY
instructblip_prompt: \"A short image description:\"
llm_prompt: \"Edit the input text, such that it is culturally relevant to COUNTRY_NAME. Keep the output text of a similar length as the input text. If it is already culturally relevant to COUNTRY_NAME, no need to make any edits. The output text must be in English only.\nInput: \" # + generated_text + Output:
task_path: ''
OPENAI_API_KEY: $OPENAI_API_KEY"

# Iterate over the countries and create config files
for index in "${!countries[@]}"
do
    # get country and country name
    country=${countries[$index]}
    country_name=${country_names[$index]}
    # source countries is a list of all countries except the current country
    source_countries=(${countries[@]})
    unset source_countries[$index]

    # Create source countries string
    source_countries_str="["
    for source_country in "${source_countries[@]}"
    do
        source_countries_str="$source_countries_str'$source_country', "
    done
    source_countries_str="${source_countries_str::-2}]"

    echo "Source countries for $country_name: $source_countries_str"

    # Create the config file
    country_config="${config_template//COUNTRY_NAME/$country_name}"
    country_config="${country_config//COUNTRY/$country}"
    country_config="${country_config//SOURCE_COUNTRIES/$source_countries_str}"

    echo "Creating config file for $country_name"
    echo "$country_config"
    # Write the config to a file
    echo "$country_config" > "$ROOT_DIR/configs/part1/caption-llm_edit/$country.yaml"
done

echo "Config files created for all countries."

