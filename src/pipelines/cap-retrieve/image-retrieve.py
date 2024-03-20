import os
from clip_retrieval.clip_back import load_clip_indices, KnnService, ClipOptions
import requests
from pathlib import Path
import json
import random
import logging
import argparse
import yaml
import pandas as pd
from PIL import Image

def download_image(image_url, folder_path, file_name=None):
    """
    Downloads an image from a given URL and saves it to a specified folder.
    Includes checks for content type, validity of the image, file size, and retry mechanism.

    :param image_url: URL of the image to download
    :param folder_path: Path to the folder where the image will be saved
    :param file_name: Name of the file (optional). If not provided, it will be derived from the URL.
    """
    try:
        if not file_name:
            file_name = image_url.split('/')[-1]
        
        # Ensure folder exists
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        response = requests.get(image_url, timeout=120)
        if response.status_code == 200 and response.headers['Content-Type'].startswith('image'):
            # Save the image
            file_path = Path(folder_path) / file_name
            with open(file_path, 'wb') as file:
                file.write(response.content)
                
            # Check for minimal file size (10KB)
            if file_path.stat().st_size < 10240:
                logging.info(f"Image saved is null or suspiciously small (size: {file_path.stat().st_size} bytes)")
                file_path.unlink()  # Remove the small file
                return "error"
                
            # Check if the image is valid
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Verify that it is, in fact, an image
            except Exception as e:
                logging.info(f"Invalid image file: {e}")
                file_path.unlink()  # Remove the corrupted file
                return "error"
            
            logging.info(f"Image saved as {file_path}")
            return "success"
        else:
            logging.info(f"Invalid response")
            return "error"
    except requests.RequestException as e:
        logging.info(f"Error downloading the image: {e}")
        return "error"
    except IOError as e:
        logging.info(f"Error saving the image: {e}")
        return "error"


def main():
    # set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"

    # read in the config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/part1/cap-retrieve/brazil.yaml", help="Path to config file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # read in the metadata file
    data = pd.read_csv(config["metadata_path"])

    src_image_paths_original = data[config["src_image_path_col"]].tolist()
    blip_captions = data[config["caption_col"]].tolist()
    llm_edits = data[config["llm_edit_col"]].tolist()
    src_countries = data[config["src_country_col"]].tolist()

    src_image_paths = src_image_paths_original.copy()
    # Add src_country to the src_image_path
    for i in range(len(src_image_paths)):
        src_country = src_countries[i]
        src_image_path_basename = src_image_paths[i].split("/")[-1]
        src_image_paths[i] = src_image_paths[i].replace(src_image_path_basename, src_country + "_" + src_image_path_basename)


    # setup clip options
    clip_options = ClipOptions(
        indice_folder = config["indice_folder"],
        clip_model = config["clip_model"],
        enable_hdf5 = False,
        enable_faiss_memory_mapping = True,
        columns_to_return = ["image_path", "caption", "url", "height", "width"],
        reorder_metadata_by_ivf_index = False,
        enable_mclip_option = False,
        use_jit = False,
        use_arrow = False,
        provide_safety_model = False,
        provide_violence_detector =  False,
        provide_aesthetic_embeddings =  False,
    )

    # load indices
    loaded_indices = load_clip_indices(config["indices_path"], clip_options)

    # construct the knn search object
    knn_service = KnnService(clip_resources=loaded_indices)

    captions = {}
    image_paths = {}
    image_urls = {}
    for prompt, src_image_path in zip(llm_edits, src_image_paths):
        logging.info(prompt)

        # get basename of the image and append country to the basename
        captions[src_image_path] = []
        image_paths[src_image_path] = []
        
        results = knn_service.query(text_input=prompt,
                                    modality="image", 
                                    indice_name=config["indice_name"], 
                                    num_images=250, 
                                    num_result_ids=250,
                                    deduplicate=True)
        
        for result in results:
            captions[src_image_path].append(result["caption"])
            image_paths[src_image_path].append(result["image_path"])
            

    logging.info(captions)
    image_urls = {}
    retrieved_captions = {}
    for src_image in captions:
        image_urls[src_image] = []
        retrieved_captions[src_image] = []
    with open(config["jsonl_file_path"], 'r') as file:
        for i, line in enumerate(file):
            # Parse each line as JSON
            json_data = json.loads(line.strip())
            og_caption = json_data["TEXT"]
            # Do something with the JSON object
            for src_image in captions:
                if og_caption in captions[src_image]:
                    idx = captions[src_image].index(og_caption)
                    if str(i) in str(image_paths[src_image][idx]):
                        height = json_data["HEIGHT"]
                        width = json_data["WIDTH"]
                        similarity = json_data["similarity"]
                        if height > 256 and width > 256 and similarity > 0.3:
                            url = json_data["URL"]
                            image_urls[src_image].append(url)
                            retrieved_captions[src_image].append(og_caption)

    random.seed(0)
    # make the output directory, first get directory name from output_file
    output_dir = config["output_file"].split("/")[:-1]
    os.makedirs("/".join(output_dir), exist_ok=True)
    with open(config["output_file"], 'w') as file:
        file.write("src_image_path,tgt_image_path,caption,llm_edit,retrieved_caption\n")
        for src_image, original_path in zip(captions, src_image_paths_original):
            logging.info(src_image)
            flag = False
            if len(image_urls[src_image]) == 0:
                logging.info("No image found for: " + src_image)
                continue
            filename = src_image.split("/")[-1]
            tgt_image_path = config["tgt_image_path"] + "/" + filename
            for url in image_urls[src_image]:
                status = download_image(url, config["tgt_image_path"], filename)
                if status == "error":
                    continue
                else:
                    flag = True
                    break
            if not flag:
                logging.info("No image found for: " + src_image)
                logging.info(len(image_urls[src_image]))
                continue
            idx = image_urls[src_image].index(url)
            retrieved_caption = retrieved_captions[src_image][idx].replace("\"", "'")
            retrieved_caption = retrieved_caption.replace(",", ";")

            file.write(original_path + "," + tgt_image_path + ",\"" + blip_captions[src_image_paths.index(src_image)] + "\",\"" + llm_edits[src_image_paths.index(src_image)] + "\",\"" + retrieved_caption + "\"\n")


if __name__ == "__main__":
    main()



