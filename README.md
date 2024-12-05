
# Plant Disease Classification with Custom ResNet-34

This repository implements a custom **ResNet-34 model** for classifying plant diseases using the [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset). The model is built from scratch in TensorFlow.
## Project Overview

Plant diseases pose a significant threat to agriculture, and early detection is crucial for mitigating crop losses. This project leverages deep learning to automate the identification of plant diseases from leaf images. The dataset contains labeled images of healthy and diseased plant leaves, and the custom ResNet-34 model is trained to achieve high classification accuracy.

## Model Architecture

The model follows the ResNet-34 architecture, implementing residual blocks with identity and convolutional shortcuts. Below is a schematic of the ResNet-34 structure:
![Model Architecture](https://github.com/user-attachments/assets/42fd5d99-8a6e-4a99-b3e8-243610c0a5f4)


### Key Features:
- **Residual Connections** to ease the training of deep networks.
- **Batch Normalization** for improved convergence.
- **Custom Layers** for adapting to the specific dataset.

## Dataset

The [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) contains over 50,000 labeled images across various plant species and disease categories. Below are sample images from the dataset:
![Dataset](https://github.com/user-attachments/assets/f7e1ebda-5437-4d93-b5e6-fadd17729be1)

## Results

The final model achieved an accuracy of **95.70%** on the test set. The training history is provided below, showing the convergence over epochs:

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/b23af68d-a934-4063-8758-dfe51d7c87a2" alt="Loss Vs Epoch" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/42f4f67e-78a8-4dab-b336-bd4aee035669" alt="Accuracy Vs Epoch" width="400"></td>
  </tr>
</table>



## References

- [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)


