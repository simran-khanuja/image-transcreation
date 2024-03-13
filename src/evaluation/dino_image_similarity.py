import os
import yaml
import argparse
import pandas as pd
import torch
from transformers import ViTImageProcessor, ViTModel
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
    parser.add_argument("--config", default="configs/dino_image_similarity.yaml", help="Path to config file.")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # mkdir config["output_dir"] if it doesn't exist
    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"], exist_ok=True)
    
    processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb8')
    model = ViTModel.from_pretrained('facebook/dino-vitb8').eval().to(device)

    data = pd.read_csv(config["metadata"])

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
            _ = processor(download_image(all_source_image_paths[i]), return_tensors="pt", padding=True).to(device)
            _ = processor(download_image(all_target_image_paths[i]), return_tensors="pt", padding=True).to(device)
        except:
            print(f"Source image path or target image path {all_target_image_paths[i]} cannot be downloaded. Removing from list.")
            continue
        source_image_paths.append(all_source_image_paths[i])
        target_image_paths.append(all_target_image_paths[i])

    batch_size = 16
    cosine_similarities = []

    with torch.no_grad():
        for i in range(0, len(source_image_paths), batch_size):
            source_images = []
            target_images = []
            for j in range(i, min(i + batch_size, len(source_image_paths))):
                source_images.append(download_image(source_image_paths[j]))
                target_images.append(download_image(target_image_paths[j]))
            source_inputs = processor(source_images, return_tensors="pt", padding=True).to(device)
            target_inputs = processor(target_images, return_tensors="pt", padding=True).to(device)
            source_features = model(**source_inputs).last_hidden_state.mean(dim=1)
            target_features = model(**target_inputs).last_hidden_state.mean(dim=1)
            source_features = torch.nn.functional.normalize(source_features, p=2, dim=1)
            target_features = torch.nn.functional.normalize(target_features, p=2, dim=1)
            cosine_similarities.extend(F.cosine_similarity(source_features, target_features).cpu().tolist())

    print(len(cosine_similarities))

    with open(config["output_dir"] + "/src_tgt_img_sim-dino.txt", "w") as f:
        for i, src_path in zip(cosine_similarities, source_image_paths):
            f.write(src_path + "," + str(i) + "\n")
   
if __name__ == "__main__":
    main()

    
    