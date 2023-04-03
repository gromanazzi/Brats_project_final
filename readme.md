# Brain Tumor Segmentation

The aim of this project is to try to segmentate and corrently predict brain tumors from **MRI** files.
To achieve this objective, it will be used Matlab with some of externali libraries available in its 

# Prerequisite
## Dataset
The first and most important task you need to manually perform is to download the [sample data](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2) and unpack it in the root folder of this project.
## Add-ons
On your MatLab release, make sure you have these add-ons installed:
 - [Deep Learning Toolbox](https://it.mathworks.com/help/deeplearning/index.html?lang=en), used to perform train and later use the 3D U-net created.
 - [Image Processing Toolbox](https://it.mathworks.com/help/images/index.html), used to read, manipulate and print the MRI volumes.
 - [Computer Vision Toolbox](https://it.mathworks.com/help/vision/index.html), used to manipulate and overlap the labels onto the MRI scans.
 - [Parallel Computing Toolbox](https://it.mathworks.com/help/parallel-computing/index.html) *(optional)* used to increase the training performance in in the training task, exploting the local GPU (if present) and providing multi-threading capability.
 ## Folder structure
 Make sure your folder structure looks like this (after the *git clone* and the DataSet download)
  - /Task01_BrainTumour (this is where your unpacked *nii.gz* files will be)
    - /imageTr
    - /imagesTs
    - /labelsTs
  - /src (source files for the project)
  - /trained_nets (pre-trained neural network ready to use)
  - main.m

# Usage
Using MatLab open *main.m* file and follow the instruction provided. There will be four options:
- **Train**: use this to train your own 3D U-Net from scratch (***Warning!*** Very high usage of PC resources and time)
-  **Segmentation**: use this option to perform a real-time segmentation choosing a volume and a trained net
-  **Performance Evaluation**: use this option to compute metrics related to a pre trained network, both on a single volume or on a test directory
- **Example**: use this option to perform a complete cycle of selecting, segmenting and printing a volume using an example net
