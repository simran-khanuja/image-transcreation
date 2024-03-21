import pandas as pd
import yaml
import argparse
import random
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/part1/collect_results/brazil.yaml", help="Path to config file")
    args = parser.parse_args()

    random.seed(0)
    # read in the config file
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # read in your data 
    e2e_instruct_df = pd.read_csv(config["e2e-instruct_metadata"])
    cap_edit_df = pd.read_csv(config["cap-edit_metadata"])
    cap_retrieve_df = pd.read_csv(config["cap-retrieve_metadata"])

    domains = []
    # src_task_paths = src_df["image_path"].tolist()
    e2e_instruct_src_paths = e2e_instruct_df["src_image_path"].tolist()
    cap_edit_src_paths = cap_edit_df["src_image_path"].tolist()
    cap_retrieve_src_paths = cap_retrieve_df["src_image_path"].tolist()
    
    e2e_instruct_tgt_paths = e2e_instruct_df["tgt_image_path"].tolist()
    cap_edit_tgt_paths = cap_edit_df["tgt_image_path"].tolist()
    cap_retrieve_tgt_paths = cap_retrieve_df["tgt_image_path"].tolist()
    # tasks = src_df["task"].tolist()

    # create dictionaries of all 3 with src_paths as the keys and tgt_paths as the values
    # task_dict = dict(zip(src_task_paths, tasks))
    e2e_instruct_dict = dict(zip(e2e_instruct_src_paths, e2e_instruct_tgt_paths))
    cap_edit_dict = dict(zip(cap_edit_src_paths, cap_edit_tgt_paths))
    cap_retrieve_dict = dict(zip(cap_retrieve_src_paths, cap_retrieve_tgt_paths))
    
    image_paths = []
    for src_path in cap_retrieve_src_paths:
        # check if the src path exists in the other two datasets
        if src_path in e2e_instruct_dict and src_path in cap_edit_dict:
            # get the tgt paths from each dataset
            e2e_instruct_tgt_path = e2e_instruct_dict[src_path]
            cap_edit_tgt_path = cap_edit_dict[src_path]
            cap_retrieve_tgt_path = cap_retrieve_dict[src_path]
            # task = task_dict[src_path]

            # shuffle the tgt paths
            tgt_paths = [e2e_instruct_tgt_path, cap_edit_tgt_path, cap_retrieve_tgt_path]
            random.shuffle(tgt_paths)
            image_paths.append([src_path, tgt_paths[0], tgt_paths[1], tgt_paths[2]])
    # shuffle the image paths
    random.shuffle(image_paths)
    for paths in image_paths:   
        src_path = paths[0]
        image_id = src_path.split("/")[-1].split(".")[0]

        # get the domains from the paths
        tgt_domains = []
        for path in paths[1:]:
            if "e2e-instruct" in path:
                tgt_domains.append("e2e-instruct")
            elif "cap-edit" in path:
                tgt_domains.append("cap-edit")
            elif "cap-retrieve" in path:
                tgt_domains.append("cap-retrieve")
        
        domains.append([image_id, tgt_domains[0], tgt_domains[1], tgt_domains[2]])

    if not os.path.exists(config["metadata_save_path"]):
        os.makedirs(config["metadata_save_path"], exist_ok=True)
    
    # can you make 5 equal splits of the data? make sure all images come in the last split
    num_splits = 5
    split_size = len(image_paths) // num_splits
    splits = []
    split_domains = []
    for i in range(num_splits-1):
        split = image_paths[i*split_size:(i+1)*split_size]
        split_domain = domains[i*split_size:(i+1)*split_size]
        splits.append(split)
        split_domains.append(split_domain)
    
    # add the remaining images to the last split
    split = image_paths[(num_splits-1)*split_size:]
    split_domain = domains[(num_splits-1)*split_size:]
    splits.append(split)
    split_domains.append(split_domain)

    for i, split in enumerate(splits):
        with open(config["metadata_save_path"]+"/split_"+str(i+1)+".csv", "w") as f:
            f.write("id,src_image_path,model_path_1,model_path_2,model_path_3,model_1,model_2,model_3\n")
            for image_path_list, domain_list in zip(split, split_domains[i]):
                src_image_path = image_path_list[0]
                model_path_1 = image_path_list[1]
                model_path_2 = image_path_list[2]
                model_path_3 = image_path_list[3]
                model_1 = domain_list[1]
                model_2 = domain_list[2]
                model_3 = domain_list[3]
                f.write(f"{src_image_path},{model_path_1},{model_path_2},{model_path_3},{model_1},{model_2},{model_3}\n")
        print("Split", i+1, "saved to", config["metadata_save_path"]+"/split_"+str(i+1)+".csv")
    # with open(config["metadata_save_path"]+"/metadata.csv", "w") as f:
    #     f.write("id,src_image_path,model_path_1,model_path_2,model_path_3,model_1,model_2,model_3\n")
    #     for image_path_list, domain_list in zip(image_paths, domains):
    #         src_image_path = image_path_list[0]
    #         model_path_1 = image_path_list[1]
    #         model_path_2 = image_path_list[2]
    #         model_path_3 = image_path_list[3]
    #         model_1 = domain_list[1]
    #         model_2 = domain_list[2]
    #         model_3 = domain_list[3]
    #         f.write(f"{src_image_path},{model_path_1},{model_path_2},{model_path_3},{model_1},{model_2},{model_3}\n")
    # print("Metadata saved to", config["metadata_save_path"]+"/metadata.csv")


if __name__ == "__main__":
    main()
    
