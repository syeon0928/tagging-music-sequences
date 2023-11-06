#!/usr/bin/env python
# coding: utf-8

import boto3
import zipfile
from zipfile import ZipFile
from pathlib import Path
import os

# Load data from AWS S3 Bucket
# This requires `aws configure` and correct credentials (Access Key ID + Secret Access Key). Alex can provide :D

# Extract s3 bucket data
s3 = boto3.client('s3')
bucket_name = 'aml-project-data-2023'

# Define your local data directory
local_data_directory = '../data'

# Zipped directories in S3 to be downloaded
s3_zipped_files = ['gtzan/gtzan.zip', 'mtat/mtat.zip']

def download_from_s3(bucket_name, s3_file, local_path):
    # Download the file from S3
    s3.download_file(bucket_name, s3_file, str(local_path))
    print(f"Downloaded {s3_file} to {local_path}")

def unzip_file(zip_path, extract_to):
    # Unzip the file into the specified directory
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Unpacked {zip_path} to {extract_to}")

# Make sure the local data directory exists
Path(local_data_directory).mkdir(parents=True, exist_ok=True)

# Main extraction
for s3_zip_file in s3_zipped_files:
    # Set up local paths
    local_zip_path = Path(local_data_directory) / Path(s3_zip_file).name
    target_unzip_directory = Path(local_data_directory) / s3_zip_file.split('/')[0]

    # Create the target directory if it doesn't exist
    target_unzip_directory.mkdir(parents=True, exist_ok=True)

    # Download zip file from S3 to local path
    download_from_s3(bucket_name, s3_zip_file, local_zip_path)

    # Unzip the file into the target directory
    unzip_file(local_zip_path, target_unzip_directory)

    # Optionally, remove the zip file after extraction
    local_zip_path.unlink()


# mtat data specific unzipping (concatenate zip files and unzip archive)
data_dir = '../data/mtat/Data'

# The path to the concatenated zip file you want to create
concatenated_zip_path = os.path.join(data_dir, 'mp3_complete.zip')

# Concatenate the split zip files into one complete zip file
with open(concatenated_zip_path, 'wb') as wfd:
    for i in range(1, 4):  # Modify range if more parts
        part_filename = f'mp3.zip.00{i}'
        part_file_path = os.path.join(data_dir, part_filename)
        with open(part_file_path, 'rb') as fd:
            wfd.write(fd.read())
            os.remove(part_file_path)  # Remove the part file after concatenation if desired

# Now you can unzip the concatenated zip file
with ZipFile(concatenated_zip_path, 'r') as zip_ref:
    zip_ref.extractall(data_dir)

# Delete the concatenated ZIP file after extraction if desired
os.remove(concatenated_zip_path)

print("All downloads and unpacking completed.")
