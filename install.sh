#!/bin/bash

# Function to check if Conda is installed
check_conda() {
  if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please install Conda first."
    exit 1
  fi
}

# Function to create and activate Conda environment
create_activate_conda_env() {
  conda env create -f environment.yml
  conda activate brain-env
}

# Function to download a file using wget
download_file() {
  url="https://pub-6129975179ce43e5b9eacdff0920180a.r2.dev/model_100_0.001_4_ResNet50_split9010.pt" # Replace with the actual URL
  output_dir="./model/"
  output_file="${output_dir}model_100_0.001_4_ResNet50_split9010.pt"

  # Create the directory if it doesn't exist
  mkdir -p "$output_dir"
  cd "$output_dir" || exit

  # Use wget to download the file
  curl --remote-name "$url"

  # Display a message after completion
  echo "File has been downloaded and saved to: $output_file"
}

# Start the installation
echo "Starting the installation..."

# # Check if Conda is installed
# check_conda

# # Create and activate the Conda environment
# create_activate_conda_env

# Download the file using wget
download_file

# Display a completion message
echo "Installation completed."
