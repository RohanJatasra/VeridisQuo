#!/bin/bash

data_dir="data"
dataset="ff-c23"
output_path="FaceForensics++_C23"
mkdir -p "$data_dir"

zip_file="$data_dir/$dataset.zip"
unzipped_dir="$data_dir/$output_path"

if [ -d "$unzipped_dir" ]; then
    echo "Data is already downloaded and extracted in $unzipped_dir."
    exit 0
fi

if [ -f "$zip_file" ]; then
    echo "Zip file already exists. Extracting..."
    unzip -o "$zip_file" -d "$data_dir"
    exit 0
fi

echo "Downloading dataset..."
kaggle datasets download xdxd003/$dataset -p "$data_dir"
echo "Extracting..."
unzip -o "$zip_file" -d "$data_dir"