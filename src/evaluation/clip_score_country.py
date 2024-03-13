import os
import yaml
import argparse
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import PIL
from PIL import Image
import torch.nn.functional as F


def download_image(path):
    image = PIL.Image.open(path)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def main():
    # Initialize device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Read in config file to get args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/clip_score_country.yaml", help="Path to config file.")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # mkdir config["output_dir"] if it doesn't exist
    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"], exist_ok=True)

    data = pd.read_csv(config["metadata"])

    # Calculate clip similarity of source and target images with prompts
    # Load CLIP model
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # Read in source image paths
    all_source_image_paths = data["src_image_path"].tolist()

    # Read in target image paths
    all_target_image_paths = data["tgt_image_path"].tolist()

    source_image_paths = []
    target_image_paths = []

    # check if paths are valid and remove paths that don't exist from both
    for i in range(len(all_source_image_paths)):
        if not os.path.exists(all_source_image_paths[i]) or not os.path.exists(all_target_image_paths[i]):
            print(f"Source image path or target image path {all_target_image_paths[i]} does not exist. Removing from list.")
            continue
        # also check if the image can be downloaded
        try:
            download_image(all_source_image_paths[i])
            download_image(all_target_image_paths[i])
            # also try processing the image
            _ = clip_processor(text=None, images=[download_image(all_source_image_paths[i])], return_tensors="pt", padding=True).to(device)
            _ = clip_processor(text=None, images=[download_image(all_target_image_paths[i])], return_tensors="pt", padding=True).to(device)
        except:
            print(f"Source image path or target image path {all_target_image_paths[i]} cannot be downloaded. Removing from list.")
            continue
        source_image_paths.append(all_source_image_paths[i])
        target_image_paths.append(all_target_image_paths[i])

    # Read in prompt
    prompts = [config["prompt"]] * len(target_image_paths)

   # Calculate source similarity
    batch_size = 16
    source_clip_similarities = []
    with torch.no_grad():
        for i in range(0, len(source_image_paths), batch_size):
            source_images_batch = []
            for j in range(i, min(i + batch_size, len(source_image_paths))):
                source_images_batch.append(download_image(source_image_paths[j]))
            prompts_batch = prompts[i:min(i + batch_size, len(source_image_paths))]
            prompt_inputs = clip_tokenizer(prompts_batch, return_tensors="pt", padding=True).to(device)
            prompt_features = clip_model.get_text_features(**prompt_inputs)
            
            source_inputs = clip_processor(text=None, images=source_images_batch, return_tensors="pt", padding=True).to(device)
            source_features = clip_model.get_image_features(**source_inputs)
            
            # Calculate similarity
            source_clip_similarities.extend(F.cosine_similarity(prompt_features, source_features).cpu().tolist())
    
    # Calculate target similarity
    batch_size = 16
    target_clip_similarities = []
    with torch.no_grad():
        for i in range(0, len(target_image_paths), batch_size):
            target_images_batch = []
            for j in range(i, min(i + batch_size, len(target_image_paths))):
                target_images_batch.append(download_image(target_image_paths[j]))
            prompts_batch = prompts[i:min(i + batch_size, len(target_image_paths))]
            prompt_inputs = clip_tokenizer(prompts_batch, return_tensors="pt", padding=True).to(device)
            prompt_features = clip_model.get_text_features(**prompt_inputs)
            
            target_inputs = clip_processor(text=None, images=target_images_batch, return_tensors="pt", padding=True).to(device)
            target_features = clip_model.get_image_features(**target_inputs)
            
            # Calculate similarity
            target_clip_similarities.extend(F.cosine_similarity(prompt_features, target_features).cpu().tolist())
    

    # Write similarities to file, also write src image paths and target image paths
    with open(os.path.join(config["output_dir"], "src_clip_sim.txt"), "w") as f:
        for path, similarity in zip(source_image_paths, source_clip_similarities):
            f.write(f"{path}\t{similarity}\n")
    
    with open(os.path.join(config["output_dir"], "tgt_clip_sim.txt"), "w") as f:
        for path, similarity in zip(target_image_paths, target_clip_similarities):
            f.write(f"{path}\t{similarity}\n")

   
if __name__ == "__main__":
    main()

    
    