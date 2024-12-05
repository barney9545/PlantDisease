
# Plant Disease Classification with Custom ResNet-34

This repository implements a custom **ResNet-34 model** for classifying plant diseases using the [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset). The model is built from scratch in TensorFlow.
## Project Overview

Plant diseases pose a significant threat to agriculture, and early detection is crucial for mitigating crop losses. This project leverages deep learning to automate the identification of plant diseases from leaf images. The dataset contains labeled images of healthy and diseased plant leaves, and the custom ResNet-34 model is trained to achieve high classification accuracy.

## Model Architecture

The model follows the ResNet-34 architecture, implementing residual blocks with identity and convolutional shortcuts. Below is a schematic of the ResNet-34 structure:
![Model Structure](<Model Architecture.png>)


### Key Features:
- **Residual Connections** to ease the training of deep networks.
- **Batch Normalization** for improved convergence.
- **Custom Layers** for adapting to the specific dataset.

## Dataset

The [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) contains over 50,000 labeled images across various plant species and disease categories. Below are sample images from the dataset:

![Sample Image 1](./images/sample1.png)
![Sample Image 2](./images/sample2.png)
![Sample Image 3](./images/sample3.png)

## Results

The final model achieved an accuracy of **95.70%** on the test set. The training history is provided below, showing the convergence over epochs:

![Loss Vs Epoch](LossvEpoch.png) ![Accuracy Vs Epoch](AccuracyvEpoch.png)



## References

- [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)


