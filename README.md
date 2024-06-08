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
## Model Outputs (Per Pipeline)

### Pipeline-1
[Brazil-Concept](https://hub.zenoml.com/project/51904131-94ed-40fa-9a0b-51bc50c68f74/InstructPix2Pix%20(brazil-part1))
[Brazil-Application](https://hub.zenoml.com/project/d410950c-48db-4680-bb3d-fa98dd283137/InstructPix2Pix%20(Brazil-part2))

[India-Concept](https://hub.zenoml.com/project/e06d954d-0237-4082-bc8c-c69af17e3f52/InstructPix2Pix%20(india-part1))
[India-Application](https://hub.zenoml.com/project/683a3077-2558-4c60-a6fb-1dc60dc611d7/InstructPix2Pix%20(India-part2))

[Japan-Concept](https://hub.zenoml.com/project/24c0c40b-9755-4419-9950-ccbb049f07f8/InstructPix2Pix%20(japan-part1))
[Japan-Application](https://hub.zenoml.com/project/7cb964de-19cd-4268-8ac4-201c63b91d91/InstructPix2Pix%20(Japan-part2))

[Nigeria-Concept](https://hub.zenoml.com/project/683b0733-9ff4-4f1f-9852-44f11632d0d2/InstructPix2Pix%20(nigeria-part1))
[Nigeria-Application](https://hub.zenoml.com/project/f75add50-f44f-47a6-972e-5b0ae4cec5bf/InstructPix2Pix%20(Nigeria-part2))

[Portugal-Concept](https://hub.zenoml.com/project/0048843b-33ac-42a3-b54d-c0924c6db123/InstructPix2Pix%20(portugal-part1))
[Portugal-Application](https://hub.zenoml.com/project/d007e2f4-66f7-49ae-92eb-0289eeb69cc8/InstructPix2Pix%20(Portugal-part2))

[Turkey-Concept](https://hub.zenoml.com/project/822bb0a0-d08b-470d-8507-785fd1b330fb/InstructPix2Pix%20(turkey-part1))
[Turkey-Application](https://hub.zenoml.com/project/b50e91ac-823a-4594-bf83-126b0e2ba9fc/InstructPix2Pix%20(Turkey-part2))

[United-States-Concept](https://hub.zenoml.com/project/bc321f67-5c12-432b-9a99-61aabad36d82/InstructPix2Pix%20(US-part1))
[United-States-Application](https://hub.zenoml.com/project/a315829f-ff08-4eeb-92a2-bef043a92acf/InstructPix2Pix%20(US-part2))

### Pipeline-2
[Brazil-Concept](https://hub.zenoml.com/project/10a0f8b3-d3d6-4b46-938c-c73f1f310cd4/Cap-LLMedit-Plugnplay%20(brazil-part1))
[Brazil-Application]

[India-Concept](https://hub.zenoml.com/project/7ff488ba-4d70-49c4-8f94-941687c4748a/Cap-LLMedit-Plugnplay%20(India-part1))
[India-Application]

[Japan-Concept](https://hub.zenoml.com/project/32ce490c-3253-4f1f-aebe-7f360e7d1b9c/Cap-LLMedit-Plugnplay%20(japan-part1))
[Japan-Application]

[Nigeria-Concept](https://hub.zenoml.com/project/c47a0438-f3af-446a-91af-e93662b91b6f/Cap-LLMedit-Plugnplay%20(nigeria-part1))
[Nigeria-Application]

[Portugal-Concept](https://hub.zenoml.com/project/51f5cad6-63d4-4159-b10f-432e9b3afa39/Cap-LLMedit-Plugnplay%20(portugal-part1))
[Portugal-Application]

[Turkey-Concept](https://hub.zenoml.com/project/1aef08d1-c1d1-4fa9-babf-15c15c8097dc/Cap-LLMedit-Plugnplay%20(turkey-part1))
[Turkey-Application]

[United-States-Concept](https://hub.zenoml.com/project/1d9b4650-8c17-4756-8336-6ac4cd3eca46/Cap-LLMedit-Plugnplay%20(united-states-part1))
[United-States-Application]

### Pipeline-3
[Brazil-Concept](https://hub.zenoml.com/project/7232cbf1-e5a6-4c52-8a37-62e93c078869/part1%20brazil%20retrieval)
[Brazil-Application](https://hub.zenoml.com/project/e6b17e18-dbf7-4c76-a18f-d50f1b52e327/part2%20brazil%20retrieval)

[India-Concept](https://hub.zenoml.com/project/3210f7ed-b2cb-4593-96de-43072b5c3b0a/part1%20india%20retrieval)
[India-Application](https://hub.zenoml.com/project/13f225d3-70e1-40b7-ac37-17257dc9ed9e/part2%20india%20retrieval)

[Japan-Concept](https://hub.zenoml.com/project/51f58190-79aa-48ec-a289-e0f3180ff8c9/part1%20japan%20retrieval)
[Japan-Application](https://hub.zenoml.com/project/bb2df18a-747a-4638-8df3-e199be9c5410/part2%20japan%20retrieval)

[Nigeria-Concept](https://hub.zenoml.com/project/d137f45f-fff8-48c2-97c3-913d3c353bae/part1%20nigeria%20retrieval)
[Nigeria-Application](https://hub.zenoml.com/project/61a75edb-1c11-43f8-8550-abea8e6c941f/part2%20nigeria%20retrieval)

[Portugal-Concept](https://hub.zenoml.com/project/fef20f21-94d2-4e45-b54b-9bb4f4264ffd/part1%20portugal%20retrieval)
[Portugal-Application](https://hub.zenoml.com/project/82622627-e216-4120-b11d-1294cb7abaeb/part2%20portugal%20retrieval)

[Turkey-Concept](https://hub.zenoml.com/project/b1d4d2db-7b9e-431a-9922-f15178357935/part1%20turkey%20retrieval)
[Turkey-Application](https://hub.zenoml.com/project/20f1ed8e-4d67-455a-8c77-f878cdf05e64/part2%20turkey%20retrieval)

[United-States-Concept](https://hub.zenoml.com/project/1a3e82e3-e107-40db-ab33-d0c111cb2ef0/part1%20united-states%20retrieval)
[United-States-Application](https://hub.zenoml.com/project/7535d825-3134-4993-8a5a-373999cd9d54/part2%20united-states%20retrieval)

## Model Outputs (Human Evaluation)
If y'all want to visualize model outputs for each part, please refer to the zeno links below. Note that the outputs were randomized for human evaluation, can you guess which pipeline each generated image is from? 😉

### Brazil
[Concept-Split1](https://hub.zenoml.com/project/f7595e88-4092-430a-9d45-55c5ad3b52a4/brazil_part1_split_1)

[Concept-Split2](https://hub.zenoml.com/project/ce44019f-391d-417f-b7a2-58d678f2703e/brazil_part1_split_2)

[Concept-Split3](https://hub.zenoml.com/project/dbb5cbfb-ae48-420e-9a35-9e9d95b95ea4/brazil_part1_split_3)

[Concept-Split4](https://hub.zenoml.com/project/8e421113-0025-4e88-829d-c38d33676177/brazil_part1_split_4)

[Concept-Split5](https://hub.zenoml.com/project/548dc681-13f6-426a-9d54-9b003fa8155b/brazil_part1_split_5)

[Application](https://hub.zenoml.com/project/8f8f055b-9039-4ed3-8f06-358ce5226f23/brazil_part2_split_1)

### India
[Concept-Split1](https://hub.zenoml.com/project/f04d325b-42b8-4aae-a365-fc9da34118c1/india_part1_split_1)

[Concept-Split2](https://hub.zenoml.com/project/f0bbf2d6-91ab-415f-a290-94acb00fd0e4/india_part1_split_2)

[Concept-Split3](https://hub.zenoml.com/project/dc33e300-4d4d-4541-91d2-ac485cfd02ee/india_part1_split_3)

[Concept-Split4](https://hub.zenoml.com/project/29de33e8-87ad-4546-8b5d-d959804a946f/india_part1_split_4)

[Concept-Split5](https://hub.zenoml.com/project/8634a759-bc13-41c3-9d4c-8fac3b05f9b7/india_part1_split_5)

[Application](https://hub.zenoml.com/project/c8de55f9-3309-4720-9578-753fe6222306/india_part2_split_1)

### Japan
[Concept-Split1](https://hub.zenoml.com/project/bb47d9b1-d032-4867-baca-c86d3147dc9e/japan_part1_split_1)

[Concept-Split2](https://hub.zenoml.com/project/2ef27b9a-0f58-40a0-8b4e-216e403ce69c/japan_part1_split_2)

[Concept-Split3](https://hub.zenoml.com/project/3170eae8-8149-4549-b4f0-264ec8e20bca/japan_part1_split_3)

[Concept-Split4](https://hub.zenoml.com/project/b6de9d4f-f35b-42ec-830e-281e433d045c/japan_part1_split_4)

[Concept-Split5](https://hub.zenoml.com/project/b232ad0b-0636-4531-9317-5e04d497b499/japan_part1_split_5)

[Application](https://hub.zenoml.com/project/ca3f5718-b5b3-4a44-a571-d0111171d799/japan_part2_split_1)

### Nigeria
[Concept-Split1](https://hub.zenoml.com/project/d7566a20-80ed-4ef1-9023-0bc35860da16/nigeria_part1_split_1)

[Concept-Split2](https://hub.zenoml.com/project/34d73e22-a897-4672-afcd-5d2c9841f7e6/nigeria_part1_split_2)

[Concept-Split3](https://hub.zenoml.com/project/64f3aaef-c819-4a6a-9a09-f9b87fa8a71b/nigeria_part1_split_3)

[Concept-Split4](https://hub.zenoml.com/project/b0022871-4cc0-4123-ab73-394c7a266bf8/nigeria_part1_split_4)

[Concept-Split5](https://hub.zenoml.com/project/b6542945-d500-43e4-bea0-1b8fa80859e3/nigeria_part1_split_5)

[Application](https://hub.zenoml.com/project/75dab001-c33b-412c-b257-fe8c08f33d2b/nigeria_part2_split_1)

### Portugal
[Concept-Split1](https://hub.zenoml.com/project/31a5fb0c-07fb-41f8-a430-7b5df9565b6d/portugal_part1_split_1)

[Concept-Split2](https://hub.zenoml.com/project/5bbb3ffc-707c-488f-bfe3-8067c590ab4f/portugal_part1_split_2)

[Concept-Split3](https://hub.zenoml.com/project/0f4361b5-de10-417d-9d41-6845a49cec9c/portugal_part1_split_3)

[Concept-Split4](https://hub.zenoml.com/project/92a3a800-78b6-4134-bc50-c2693a444d12/portugal_part1_split_4)

[Concept-Split5](https://hub.zenoml.com/project/9073064b-384f-44e8-b5e9-ee32da794d34/portugal_part1_split_5)

[Application](https://hub.zenoml.com/project/6fa064e5-80ce-46d0-8b60-65879718eb79/portugal_part2_split_1)

### Turkey
[Concept-Split1](https://hub.zenoml.com/project/5e8c6aaa-1a07-42d7-866c-85071c666ba5/turkey_part1_split_1)

[Concept-Split2](https://hub.zenoml.com/project/b00cc8eb-fbc5-428b-bdd0-896438a42795/turkey_part1_split_2)

[Concept-Split3](https://hub.zenoml.com/project/a08b0a37-6a09-4cbf-b8a0-d5060eb92601/turkey_part1_split_3)

[Concept-Split4](https://hub.zenoml.com/project/82ce07a0-f52d-4f95-9d9a-34b0695b46df/turkey_part1_split_4)

[Concept-Split5](https://hub.zenoml.com/project/c7c287a2-d48d-4ed3-9812-88567bc6bc6a/turkey_part1_split_5)

[Application](https://hub.zenoml.com/project/53b33bce-56fb-4fbd-8445-f1eee87a39cc/turkey_part2_split_1)

### United States
[Concept-Split1](https://hub.zenoml.com/project/728c9bb1-58ce-4089-9685-287fa1d4d3e9/united-states_part1_split_1)

[Concept-Split2](https://hub.zenoml.com/project/5974f9d1-f9d7-49d8-a68b-e67d1ff1c0d1/united-states_part1_split_2)

[Concept-Split3](https://hub.zenoml.com/project/320643b3-090b-4eef-87c6-94ed0c4e6e9a/united-states_part1_split_3)

[Concept-Split4](https://hub.zenoml.com/project/cad5995e-503b-4cf5-847d-0304a12a48ea/united-states_part1_split_4)

[Concept-Split5](https://hub.zenoml.com/project/917f7000-6a6e-4894-b7e5-f95511aa826d/united-states_part1_split_5)

[Application](https://hub.zenoml.com/project/f3090a24-8958-4a5e-930e-4a32353d5d94/united-states_part2_split_1)



