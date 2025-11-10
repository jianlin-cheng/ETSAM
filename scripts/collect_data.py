import argparse
import os
import pandas as pd
from cryoet_data_portal import Client, Run
from pyDataverse.api import NativeApi, DataAccessApi
import shutil
import requests
from tqdm import tqdm

# Instantiate a client, using the data portal GraphQL API by default
client = Client()

HARVARD_DATAVERSE_SERVER_URL = "https://dataverse.harvard.edu"
ETSAM_DATASET_PID = "doi:10.7910/DVN/K4JKCW"

harvard_native_api = NativeApi(base_url=HARVARD_DATAVERSE_SERVER_URL)
harvard_data_api = DataAccessApi(base_url=HARVARD_DATAVERSE_SERVER_URL)
harvard_dataset = harvard_native_api.get_dataset(ETSAM_DATASET_PID)
harvard_dataset_files = harvard_dataset.json()['data']['latestVersion']['files']

def get_harvard_file_name_and_id(dataset_id, run_id, file_type="target_mask"):
    file_dir = os.path.join(dataset_id, run_id, file_type)
    for file in harvard_dataset_files:
        if file['directoryLabel'] == file_dir:
            return file['label'], file['dataFile']['id']
    return None

def download_https(
    url: str,
    dest_path: str,
    with_progress: bool = True,
):
    fetch_request = requests.get(url, stream=True)
    total_size = int(fetch_request.headers["content-length"])
    block_size = 1024 * 512
    with tqdm(
        total=total_size,
        unit="iB",
        unit_scale=True,
        disable=(not with_progress),
    ) as progress_bar, open(dest_path, "wb") as f:
        for data in fetch_request.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)


def download_experimental_tomogram(dataset_id, run_id, run_dir):
    # download the experimental tomogram from CDP
    print(f"--> Downloading tomogram for Dataset: {dataset_id} Run: {run_id}")
    tomogram_dir = os.path.join(run_dir, "tomogram")
    os.makedirs(tomogram_dir, exist_ok=True)

    tomogram = Run.find(client, [Run.id == run_id])[0].tomograms[0]
    tomogram.download_mrcfile(dest_path=tomogram_dir)

    # download the target mask from Harvard Dataverse
    print(f"--> Downloading target mask for Dataset: {dataset_id} Run: {run_id}")
    target_mask_dir = os.path.join(run_dir, "target_mask")
    os.makedirs(target_mask_dir, exist_ok=True)

    file_name, file_id = get_harvard_file_name_and_id(dataset_id, run_id, file_type="target_mask")
    if file_id is None:
        print(f"Error: No target mask file found for Dataset: {dataset_id} Run: {run_id}")
        exit()

    target_mask_file = os.path.join(target_mask_dir, file_name)
    download_https(f"{HARVARD_DATAVERSE_SERVER_URL}/api/access/datafile/{file_id}", target_mask_file)

def download_simulated_tomogram(dataset_id, run_id, run_dir):
    # download the tomogram from Harvard Dataverse
    print(f"--> Downloading tomogram for Dataset: {dataset_id} Run: {run_id}")
    tomogram_dir = os.path.join(run_dir, "tomogram")
    os.makedirs(tomogram_dir, exist_ok=True)

    file_name, file_id = get_harvard_file_name_and_id(dataset_id, run_id, file_type="tomogram")
    if file_id is None:
        print(f"Error: No target mask file found for Dataset: {dataset_id} Run: {run_id}")
        exit()

    tomogram_file = os.path.join(tomogram_dir, file_name)
    download_https(f"{HARVARD_DATAVERSE_SERVER_URL}/api/access/datafile/{file_id}", tomogram_file)

    # download the target mask from Harvard Dataverse
    print(f"--> Downloading target mask for Dataset: {dataset_id} Run: {run_id}")
    target_mask_dir = os.path.join(run_dir, "target_mask")
    os.makedirs(target_mask_dir, exist_ok=True)

    file_name, file_id = get_harvard_file_name_and_id(dataset_id, run_id, file_type="target_mask")
    if file_id is None:
        print(f"Error: No target mask file found for Dataset: {dataset_id} Run: {run_id}")
        exit()

    target_mask_file = os.path.join(target_mask_dir, file_name)
    download_https(f"{HARVARD_DATAVERSE_SERVER_URL}/api/access/datafile/{file_id}", target_mask_file)

def main():
    parser = argparse.ArgumentParser(description="Collect data used in ETSAM.")
    parser.add_argument("--csv", type=str, default="data/dataset.csv", help="Path to the input CSV file")
    parser.add_argument("--output-dir", type=str, default="data/collection", help="Directory to save the output")

    args = parser.parse_args()
    csv_file = args.csv
    output_dir = args.output_dir
    status_dir = os.path.join(output_dir, ".status")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(status_dir, exist_ok=True)

    print("Takes long time to download data => Rerun to resume if interrupted in the middle.")
    print("========================================================================================")
    df = pd.read_csv(csv_file, dtype={'dataset_id': str, 'run_id': str})
    df = df[(df["split"] == "train") | (df["split"] == "test")].reset_index(drop=True)
    
    for index, row in df.iterrows():
        dataset_id = row['dataset_id']
        run_id = row['run_id']

        status_file = os.path.join(status_dir, f"{dataset_id}_{run_id}_completed.txt")
        if os.path.exists(status_file):
            print(f"==> {index + 1} of {len(df)} - Skipping Dataset: {dataset_id} Run: {run_id} because it has already been completed")
            continue

        run_dir = os.path.join(output_dir, dataset_id, run_id)
        shutil.rmtree(run_dir, ignore_errors=True)
        os.makedirs(run_dir, exist_ok=True)
        
        print(f"==> {index + 1} of {len(df)} - Fetching Dataset: {dataset_id} Run: {run_id}")

        if "polnet" in dataset_id:
            download_simulated_tomogram(dataset_id, run_id, run_dir)
        else:
            download_experimental_tomogram(dataset_id, run_id, run_dir)
    
        # touch the status file
        with open(status_file, "w") as f:
            f.write("Completed")

    
    print("========================================================================================")
    print("All done!")
    print("========================================================================================")


if __name__ == "__main__":
    main()

