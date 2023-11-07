#!/usr/bin/env python
# coding: utf-8
import boto3
from pathlib import Path
from tqdm import tqdm

# Load data from AWS S3 Bucket
# This requires `aws configure` and correct credentials (Access Key ID + Secret Access Key).
s3 = boto3.client('s3')
bucket_name = 'aml-project-data-2023'

# Define local data directory relative to the script location
local_data_directory = Path(__file__).parent.parent / 'data' / 'raw'

# Directories in S3 to be downloaded
s3_directories = ['gtzan_data/', 'mtat_data/']

def download_s3_directory(bucket_name, s3_folder, local_path):
    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder):
        # Filter out the directory name
        if 'Contents' in result:  # check if 'Contents' key is present
            files = [obj['Key'] for obj in result['Contents'] if not obj['Key'].endswith('/')]
            for file in tqdm(files, desc=f"Downloading {s3_folder}"):
                # Define the destination path and create any intermediate directories
                destination_path = local_path / Path(file).relative_to(s3_folder)
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                # Download the file
                s3.download_file(bucket_name, file, str(destination_path))

# Ensure the local data directory exists
Path(local_data_directory).mkdir(parents=True, exist_ok=True)

# Download specified S3 directories to local paths
for directory in s3_directories:
    directory_path = Path(local_data_directory) / directory.strip('/')
    directory_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    download_s3_directory(bucket_name, directory, directory_path)

print("All downloads completed.")
