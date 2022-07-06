# A comparison of deep learning thermography-based methods for the state assessment of rolling bearings

Code for Hybrid and non Hybrid convolutional models (based on EfficientNet and VGG19) for the state assessment of rolling bearings using infrared thermography.

## Hybrid Model

For the hybrid model we tested two types of input data:
- Concatenating thermal pixel matrix to the thermal images and use this 4-channel data as input to the CNN.
- Using original thermal image as input to the convolutional backbone and use feature vector from temperature data as input to the linear classification layers.

## main.py

We added a file main.py which allow to reproduce the main experiments of our paper and perform classification over an image of your choice.

### test
In order to reproduce the experiments of our paper run the following command:
```
python main.py --mode test

```
This will reproduce our best performance approach, a feature vector hybrid model with bicubic interpolated thermal matrix. However, the ```--experiment``` argument will allow you to reproduce other experiments from the paper. Use as input to this argument the notation on the paper, e.g., to reproduce the 4 channel hybrid model with no resizing and zero padded thermal matrix use:

```
python main.py --mode test --experiment 4-HynoRS-interpol
```

### demo

To use our best model to perform classification over an image of your choice, choose an image from our dataset, e.g. *IR000016.png*, and run the following command:

```
python main.py --mode demo --img IR000016
```
That's correct, please input the image name without extension, since we are using a hybrid model and .png and .csv files are used. Demo mode also accept ```--experiment``` argument to use weights from other experiments for IRT classification.
