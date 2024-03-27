from PIL import Image
import os
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
import PIL
import yaml
import argparse
import pandas as pd
import logging
import openai
import json
import requests
from io import BytesIO

INSTRUCT_PROMPT = {
    "education": "This image is from a math worksheet titled: TASK. Describe the image such that it talks about details relevant to the task of the worksheet. The output should be ONLY ONE sentence long.",
    "stories": "This image is from a storybook for children. Caption the image such that it describes details relevant to the story."
}

LLM_PROMPT = {
    "education": "The text describes an image in a math worksheet titled: TASK. Hence, make sure the edit preserves the intent of the task in the worksheet. Keep the output text to be of a similar length as the input text. If it is already culturally relevant to Brazil, there is no need to make any edits. The output text must be in English only.\nInput: ", # + generated_text + Output:
    "stories": "The text describes an image in a storybook for children. Make sure the edit preserves the meaning of the story. Keep the output text to be of a similar length as the input text. If it is already culturally relevant to Brazil, there is no need to make any edits. The output text must be in English only.\nInput: " # + generated_text + Output:"
}

def download_image(path):
    # check if image is URL and download if yes
    if path.startswith("http"):
        response = requests.get(path, timeout=120)
        if response.status_code == 200 and response.headers['Content-Type'].startswith('image'):
            image = PIL.Image.open(BytesIO(response.content))
            image = PIL.ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            return image
        else:
            logging.info(f"Invalid response")
            return "error"
    image = PIL.Image.open(path)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def answer_fn(prompt):
    response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=0.5,
            frequency_penalty=0,
            presence_penalty=0,
            n=1,
            logprobs=1,
        )

    return response.choices[0]['text'].strip()

def main():
    # Initialize device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Read in config file to get args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/part2/caption-llm_edit/brazil.yaml", help="Path to config file.")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    openai.api_key = config["OPENAI_API_KEY"]
    
    # mkdir config["output_dir"] if it doesn't exist
    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"], exist_ok=True)

    # # Initialize InstructBlip model
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xxl")
    model = InstructBlipForConditionalGeneration.from_pretrained(
        "Salesforce/instructblip-flan-t5-xxl", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.bfloat16
    )

    # Get source countries
    source_countries = config["source_countries"]
    source_data_path = config["source_data_path"]
    task_path = config["task_path"]

    all_source_paths = []
    all_source_countries = []
    for country in source_countries:
        country_paths_file = source_data_path + "/" + country + ".json"
        with open(country_paths_file) as f:
            data = json.load(f)
            # get values from json file which is a dictionary of dictionaries
            for category in data:
                all_source_paths.extend(data[category].values())
                all_source_countries.extend([country] * len(data[category].values()))
    
    print(len(all_source_paths))
    print(len(all_source_countries))
    instructblip_prompts = []
    llm_prompts = []
    if task_path != '':
        tasks = json.load(open(task_path))
        for domain in tasks:
            domain_tasks = tasks[domain]
            for key in domain_tasks:
                flag=False
                for path in all_source_paths:
                    filename = path.split("/")[-1].split(".")[0]
                    if key==filename:
                        final_instruct_prompt = INSTRUCT_PROMPT[domain].replace("TASK", "\""+domain_tasks[key]+"\"")
                        final_llm_prompt = config["llm_prompt"] + " " + LLM_PROMPT[domain].replace("TASK", "\""+domain_tasks[key]+"\"")
                        instructblip_prompts.append(final_instruct_prompt)
                        llm_prompts.append(final_llm_prompt)
                        flag=True
                        break
                if not flag:
                    print(key)
        print(len(instructblip_prompts))
        print(len(llm_prompts))
        # print((all_source_paths))
        print(instructblip_prompts[-5:])
        print(llm_prompts[-5:])
    else:
        tasks = [""] * len(all_source_paths)
        instructblip_prompts = [config["instructblip_prompt"]] * len(all_source_paths)
        llm_prompts = [config["llm_prompt"]] * len(all_source_paths)

    logging.info("Number of images: " + str(len(all_source_paths)))
    batch_size = 8
    
    blip_captions = []
    # Run the pipeline in batches and save the outputs, also save intermediate generated text
    for i in range(0, len(all_source_paths), batch_size):
        logging.info(i)
        max_index = min(i + batch_size, len(all_source_paths))
        batch = all_source_paths[i : max_index]
        images = [download_image(path) for path in batch]
        instructblip_batch = [instructblip_prompts[i] for i in range(i, max_index)]
            
        inputs = processor(images=images, text=instructblip_batch, return_tensors="pt", padding=True).to(device, torch.float16)
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=128,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=0.9,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        for text in generated_text:
            blip_captions.append(text.strip().split(".")[0])

    logging.info(blip_captions)

    # Generate LLM text
    llm_text= []
    for caption, llm_prompt in zip(blip_captions, llm_prompts):
        # llm_prompt = LLM_PROMPT.replace("\{task\}", "\""+task+"\"")
        gen_text = answer_fn(llm_prompt + caption + "\nOutput: ")
        for char in ["\"", ";", "."]:
            gen_text = gen_text.replace(char, "")
        llm_text.append(gen_text)
    
    logging.info(llm_text)
    
    # Write caption and LLM text to file
    with open(config["output_dir"] + "/metadata.csv", "w", encoding='utf-8') as f:
        f.write("src_image_path,src_country,caption,llm_edit\n")
        for i in range(len(all_source_paths)):
            blip_cp = blip_captions[i].replace("\"", "\'")
            llm = llm_text[i].replace("\"", "\'")
            f.write(all_source_paths[i] + ",\"" + all_source_countries[i] + "\",\"" + blip_cp + "\",\"" + llm + "\"\n")


if __name__ == "__main__":
    main()




