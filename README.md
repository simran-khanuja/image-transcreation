# Image Transcreation

## Setup
Create conda environment using `environment.yml`. Python version used is `Python 3.10.12`.

## Code for running image transcreation pipelines

## e2e-instruct
Run `./scripts/part1/e2e-instruct.sh` for running this pipeline for all countries.

## cap-edit
#### Step 1: Get image captions and edit them using GPT-3.5
Enter OPENAI_API_KEY in `./configs/part1/caption-llm_edit/make_configs.sh`
Run `./configs/part1/caption-llm_edit/make_configs.sh` to make config files for each country.
Run `./scripts/part1/caption-llm_edit.sh` to get captions and LLM edits for all countries using InstructBLIP and GPT-3.5 respectively.

#### Step 2: Edit images using LLM-edits and PlugnPlay
We've made modifications on top of `https://github.com/MichalGeyer/pnp-diffusers`. Kindly clone the fork of this repository from `<enter fork URL post review>` under `./src/pipelines/cap-edit/image-edit`. Follow through their readme and first create the `pnp-diffusers` environment. Image-editing using the plugnplay model involves two stages: a) obtain the noisy latents if the original image; and b) image-editing as per text guidance. To obtain latents, run the following:
```
bash ./scripts/part1/step1_pnp_preprocess.sh
```
To edit images according using the LLM edits as text guidance, run the following:
```
bash ./scripts/part1/step2_pnp_img-edit.sh
```


## cap-retrieve
#### Step 1: Get image captions and edit them using GPT-3.5
Same as for `cap-edit`. No need to run anything here if already run for `cap-edit`, else follow instructions from above to get captions and LLM edits

#### Step 2: Retrieve images from LAION-{COUNTRY} using LLM-edits as text queries
Create a fresh environment by running:
```
conda create -n clip-ret-env python=3.10
```
We leverage the [clip-retrieval](https://github.com/rom1504/clip-retrieval) infrastructure to obtain LAION indices in a scalable and efficient way. Hence, run `pip install clip-retrieval`. 

We create autofaiss indices for country specific subsets of LAION. For this, navigate to `./src/pipelines/cap-retrieve/prepare_laion` and run `categorize_cctld.py` to create json files of image paths for each country. Next follow through `step1-img2dataset.sh`, `step2-embeddings.sh` and `step3-index.sh` to create datasets from images, get embeddings for images and text, and create indices from the embeddings, respectively. 

Now, to retrieve images from LAION given a text query (here, this is the LLM-edited captions obtained in Step-1), run the following:
```
bash ./scripts/part1/cap-retrieve.sh
```

## Model Outputs (Zeno Links)
If y'all want to visualize model outputs for each part, please refer to the zeno links below. Note that the outputs were randomized for human evaluation, can you guess which pipeline each generated image is from? 😉

### Brazil
[Part1-Split1](https://hub.zenoml.com/project/f7595e88-4092-430a-9d45-55c5ad3b52a4/brazil_part1_split_1)

[Part1-Split2](https://hub.zenoml.com/project/ce44019f-391d-417f-b7a2-58d678f2703e/brazil_part1_split_2)