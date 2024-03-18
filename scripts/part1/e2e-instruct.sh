#!/bin/bash
countries=("brazil" "india" "japan" "nigeria" "portugal" "turkey" "united-states")

for country in "${countries[@]}"
do
    python ./src/pipelines/e2e-instruct.py --config ./configs/part1/e2e-instruct/$country.yaml
done