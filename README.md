# Scene segmentation task
Related paper: CVPR2020: A Local-to-Global Approach to Multi-modal Movie Scene Segmentation. 

### Requirements
- Python 3.6
- PyTorch 1.8

### Explanation

A naive approach is implemented in SceneSeg.py:

1. Model: I use two-layer FC model as an example to show the whole pipeline. 
2. Training and Test: 80% and 20%.

After putting the .pkl files in the data folder, one can just run SceneSeg.py to obtain the final mAP and mean Miou values as required. Due to limited time and computation resources, the naive approach only achieves a slighly better performance than random guess (8.3 as in the paper and 9.9 by the naive model and just 1 epoch training).  

With more time and computation resources, one can improve the model from several aspects:
1. Adjust the number of layers and hidden units in each layer.
2. Adjust other hyperparameters such as learning rate, epochs, activation function, etc.
3. Considering more factors in the modeling, e.g., temporal dependency between different shots, etc.
4. Implement related models as indicated in the CVPR2020 paper.





