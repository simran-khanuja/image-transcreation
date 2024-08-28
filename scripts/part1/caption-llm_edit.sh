#!/bin/bash

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
    python ./src/pipelines/caption-llm_edit.py --config ./configs/part1/caption-llm_edit/$country.yaml
done
