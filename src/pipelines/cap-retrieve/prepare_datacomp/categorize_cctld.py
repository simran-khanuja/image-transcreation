from urllib.parse import urlparse
from collections import defaultdict
import csv
import json
from datasets import load_dataset
from torch.utils.data import DataLoader
import logging
import argparse
import yaml
from huggingface_hub.utils._errors import HfHubHTTPError
import time
from joblib import Parallel, delayed
from itertools import islice
import os


logging.basicConfig(level=logging.INFO)

MAX_RETRIES = 20
RETRY_WAIT = 15  # wait 10 seconds before retry
BUFFER_SIZE = 50000  # Adjust this based on available memory

def extract_domain(url):
    parsed_uri = urlparse(url)
    domain = '{uri.netloc}'.format(uri=parsed_uri)
    return domain

def categorize_url_by_ccTLD(url, ccTLDs):
    domain = extract_domain(url)
    tld = domain.split('.')[-1]
    if tld in ccTLDs:
        return ccTLDs[tld]
    else:
        return 'Other'

def process_single_item(item, ccTLDs):
    url = item['url']
    country = categorize_url_by_ccTLD(url, ccTLDs)
    
    example = {}
    for key in item:
        value = item[key]
        if hasattr(value, 'tolist'):  # Check if it's a tensor
            value = value.tolist()  # Convert tensor to list
        example[key] = value
    
    return country, example

def process_batch(batch, config, ccTLDs):
    results = Parallel(n_jobs=config["num_joblib_workers"])(delayed(process_single_item)(item, ccTLDs) for item in batch)
    
    # Aggregate results
    aggregated_results = defaultdict(list)
    for country, example in results:
        if country != 'Other':
            aggregated_results[country].append(example)
    
    return aggregated_results

if __name__ == '__main__':
    
    # Read in config file to get args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/categorize_cctld.yaml", help="Path to config file.")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Make sure save path exists
    if not os.path.exists(config["save_path"]):
        os.makedirs(config["save_path"], exist_ok=True)
    
    # change save path to sav path + "/jsonl"
    config["save_path"] = config["save_path"] + "/jsonl"

    # check if save path exists, delete it and remake if it does
    if os.path.exists(config["save_path"]):
        os.system(f"rm -rf {config['save_path']}")
    os.makedirs(config["save_path"], exist_ok=True)

    iterable_dataset = load_dataset(config["dataset"], streaming=True, split='train')

    def custom_collate_fn(batch):
        new_batch = []
        for item in batch:
            new_item = {}
            for key, value in item.items():
                if value is None:
                    if key in ['original_height', 'original_width', 'clip_l14_similarity_score', 'uid']:
                        new_item[key] = 0
                    else:
                        new_item[key] = ""
                else:
                    new_item[key] = value
            new_batch.append(new_item)
        return new_batch

    dataloader = DataLoader(iterable_dataset, batch_size=config["batch_size"], collate_fn=custom_collate_fn, num_workers=config["num_dataset_workers"])

    ccTLDs = {}
    with open(config["cctld_path"], 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            key = row[0][1:]
            value = row[1]
            ccTLDs[key] = value

    dataset_stats = defaultdict(int)
    last_successful_batch = -1
    
    dataset_stats = defaultdict(int)
    buffers = defaultdict(list)
    start_time = time.time()

    for retry in range(MAX_RETRIES):
        try:
            # Using islice to start from the batch after the last_successful_batch
            data_iter = islice(enumerate(dataloader), last_successful_batch + 1, None)
            
            for j, batch in data_iter:
                if j % 100 == 0:
                    logging.info(f"Processed {j} batches in {time.time() - start_time} seconds.")
                
                local_dataset_dict = process_batch(batch, config, ccTLDs)
                for country, examples in local_dataset_dict.items():
                    dataset_stats[country] += len(examples)
                    buffers[country].extend(examples)

                    if len(buffers[country]) >= BUFFER_SIZE:
                        save_path = config["save_path"] + f"/{country}_dataset.jsonl"
                        with open(save_path, 'a') as file:
                            for example in buffers[country]:
                                json.dump(example, file)
                                file.write('\n')
                        buffers[country] = []

                last_successful_batch = j

            for country, buffer_examples in buffers.items():
                save_path = config["save_path"] + f"/{country}_dataset.jsonl"
                with open(save_path, 'a') as file:
                    for example in buffer_examples:
                        json.dump(example, file)
                        file.write('\n')

            break
        
        # except all errors and retry
        except Exception as e:
            if retry < MAX_RETRIES - 1:
                logging.warning(f"Encountered error on batch {j}. Waiting for {RETRY_WAIT} seconds before retrying.")
                time.sleep(RETRY_WAIT)
                continue
            else:
                raise
        

    # Sort dataset stats by count
    dataset_stats = {k: v for k, v in sorted(dataset_stats.items(), key=lambda item: item[1], reverse=True)}

    with open(config["dataset_stats_path"], 'w') as file:
        for country, count in dataset_stats.items():
            file.write(f"{country} {count}\n")
