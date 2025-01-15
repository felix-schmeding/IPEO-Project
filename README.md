
# Canopy Height Regression from Sentinel-2 Images
This project aims to perform per-pixel canopy height regression using Sentinel-2 imagery.

## Setup Instructions
To recreate the environment used to run this project, execute the following command:
```
conda env create -f environment.yml
```

## GitHub repository

To use the code, first clone the following repo: [IPEO Project](https://github.com/felix-schmeding/IPEO-Project) \
Then follow the next steps to ensure the data is correctly downloaded

## Dataset and File Structure
Before running the Jupyter notebook scripts, ensure you have the following file structure in place:

* **Dataset:** Extract the dataset into `./data/dataset/`
* **Model Weights & Stats:** Place the model weights and trained statistics in `./data/model/`
* **Sample Image and Label:** Put the sample image and corresponding label in `./data/sample/`

## Dataset
You can download the dataset from the following link:

[Canopy Height Dataset](https://enacshare.epfl.ch/bY2wS5TcA4CefGks7NtXg). (File name: `canopy_height_dataset.zip`)
After downloading, extract the contents to the `./data/dataset/` directory.

## Model Weights
You can download the model weights, train statistics and the sample image used in our report from this [Google Drive](https://drive.google.com/drive/folders/1RMOEC3amS9FXChPwZZRPjc5IenxynuYH?usp=sharing).

## Running the Notebooks
Once the environment is set up and the necessary files are in place, you can run the following Jupyter notebook scripts:

* **evaluation.ipynb:** Evaluates the model on different splits and tests metrics.
* **training.ipynb:** Trains the model using the dataset.
* **inference.ipynb:** Runs inference on a sample image and label.
