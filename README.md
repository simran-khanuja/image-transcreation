# Image Transcreation

## Setup
Create conda environment using `environment.yml`. Python version used is `Python 3.10.12`.

## Code for running image transcreation pipelines

### e2e-instruct
Run `./scripts/part1/e2e-instruct.sh` for running this pipeline for all countries.

### cap-edit
#### Step 1: Get image captions and edit them using GPT-3.5
Enter OPENAI_API_KEY in `./configs/part1/caption-llm_edit/make_configs.sh`
Run `./configs/part1/caption-llm_edit/make_configs.sh` to make config files for each country.
Run `./scripts/part1/caption-llm_edit.sh` to get captions and LLM edits for all countries using InstructBLIP and GPT-3.5 respectively.

#### Step 2: Edit images using LLM-edits and PlugnPlay
We've made modifications on top of `https://github.com/MichalGeyer/pnp-diffusers`. Kindly clone the fork of this repository from `<enter fork URL post review>` under `./src/pipelines/cap-edit/image-edit`. First create the `pnp-diffusers` environment. 


### cap-retrieve
#### Step 1: Get image captions and edit them using GPT-3.5
Same as for `cap-edit`. No need to run anything here if already run for `cap-edit`, else follow instructions from above to get captions and LLM edits

#### Step 2: Retrieve images from LAION-{COUNTRY} using LLM-edits as text queries
First create autofaiss indices for country specific subsets of LAION. For this, first run `./src/pipelines/cap-retrieve/prepare_laion/categorize_cctld.py` to create json files of image paths for each country. Next follow through `./src/pipelines/cap-retrieve/prepare_laion/step1-img2dataset.sh`, `./src/pipelines/cap-retrieve/prepare_laion/step2-embeddings.sh` and `./src/pipelines/cap-retrieve/prepare_laion/step3-index.sh` to create datasets from images, get embeddings for images and text, and create indices from the embeddings, respectively. We leverage the [clip-retrieval](https://github.com/rom1504/clip-retrieval) infrastructure to obtain these indices in a scalable and efficient way.

