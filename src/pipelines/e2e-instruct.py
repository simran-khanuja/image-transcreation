from PIL import Image, ImageOps
import os
import torch
import PIL
import argparse
import yaml
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import random
import logging
import pandas as pd

def download_image(path):
    image = PIL.Image.open(path)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def concatenate_images(source_img, target_img, save_path):
    total_width = source_img.width + target_img.width
    max_height = max(source_img.height, target_img.height)

    concatenated_img = Image.new('RGB', (total_width, max_height))
    concatenated_img.paste(source_img, (0, 0))
    concatenated_img.paste(target_img, (source_img.width, 0))
    concatenated_img.save(save_path)


def resize_image(image, threshold_size=1024):
    w, h = image.size
    if w > threshold_size or h > threshold_size:
        if w > h:
            new_w = threshold_size
            new_h = int(h * (threshold_size / w))
        else:
            new_h = threshold_size
            new_w = int(w * (threshold_size / h))
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return image



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
    parser.add_argument("--config", default="configs/part1/e2e-instruct/brazil.yaml", help="Path to config file.")
   
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    image_path_column = config["image_path_column"]
    
    random.seed(config["seed"])
    
    # mkdir config["output_dir"] if it doesn't exist
    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"], exist_ok=True)

    # Initialize Instruct-Pix2Pix
    instruct_pix2pix_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(instruct_pix2pix_id, torch_dtype=torch.float16, safety_checker=None)
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    # Get multiple image paths from csv file
    data = pd.read_csv(config["input_file"])
    all_image_paths = data[image_path_column].tolist()
    if args.debug:
        all_image_paths = random.sample(all_image_paths, 20)
    logging.info("Number of images: " + str(len(all_image_paths)))
    
    # Iterate over each image path and remove it if it doesn't exist
    image_paths = []
    for i in range(len(all_image_paths)):
        if os.path.exists(all_image_paths[i]):
            image_paths.append(all_image_paths[i])
    logging.info("Number of images: " + str(len(image_paths)))

    num_inference_steps = int(args.num_inference_steps)
    image_guidance_scale = float(args.image_guidance)
    guidance_scale = float(args.text_guidance)

    output_dir = config["output_dir"] + "/" + str(num_inference_steps) + "_" + str(image_guidance_scale) + "_" + str(guidance_scale)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # Generate images
    with open(output_dir + "/metadata.csv", "w") as f:
        f.write("src_image_path,tgt_image_path,prompt\n")
        for i, image_path in enumerate(image_paths):
            try:
                image = download_image(image_path)
                image = resize_image(image)
                prompt = config["prompt"]
                generated_image = pipe(prompt, image=image, num_inference_steps=num_inference_steps, image_guidance_scale=image_guidance_scale, guidance_scale=guidance_scale).images[0]
                generated_image_path = output_dir + "/" + image_path.split("/")[-1]
                generated_image.save(generated_image_path)
                f.write(image_path + "," + generated_image_path + "," + prompt + "\n")
            except torch.cuda.OutOfMemoryError as e:
                logging.info(f"Skipping image {image_path} due to CUDA OOM error: {e}")
                continue

if __name__ == "__main__":
    main()
