# Temperature based and intensity based analysis of infrared images for the state assessment of rolling bearings using Deep learning methods

Pytorch implementation of the convolutional neural network used for this paper

## Proposed architecture

In this paper we evaluate the which type of data from IRT is best for image classification. Infrared Thermography has two types of data: thermal pixel matrix and thermal images. For evaluation we use EfficientNet and DenseNet, first using thermal pixel matrix and thermal images as inputs, and then we propose a hybrid model that combines two types of data.

## Hybrid Model

For the hybrid model we tested two types of input data:
- Resize original image and use thermal pixel matrix as an additional channel.
- Using original image as input to the convolutional backbone and use feature vector from the thermal pixel matrix as input to the linear classification layers
