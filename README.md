# Scene segmentation task
Related paper: A Local-to-Global Approach to Multi-modal Movie Scene Segmentation. 

### Requirements
- Python 3.6+
- PyTorch 1.0 or higher

### Explanation

It is a self-made solution. Using two-layer FC model as an example to show the whole pipeline. One can just run SceneSeg.py to obtain the final mAP and mean Miou values as required. Due to limited time and computation resources, the naive approach only achieves a slighly better performance than random guess (8.3 as in the paper and 9.9 by the naive model).  

With more time and computation resources, one can improve the model from several aspects:
1. Adjust the number of layers and hidden units in each layer.
2. Adjust other hyperparameters such as learning rate, epochs, activation function, etc.
3. Implement related models as indicated in the CVPR2020 papers.





