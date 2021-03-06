# Diabetic-Retinopathy-Detection

## Motivation
Diabetic retinopathy is the leading cause of blindness in the working-age population of the developed world. The condition is estimated to affect over 93 million people.

The need for a comprehensive and automated method of diabetic retinopathy screening has long been recognized, and previous efforts have made good progress using image classification, pattern recognition, and machine learning. With photos of eyes as input, the goal of this capstone is to create a new model, ideally resulting in realistic clinical potential.

* Computer Vision image classification has been a personal interest for years, in addition to classification on a large scale data set.
* Time is lost between patients getting their eyes scanned, having their images analyzed by doctors, and scheduling a follow-up appointment. By processing images in real-time, **ResNet50 model** would allow people to seek & schedule treatment the same day.

## The Dataset
The data originates from a [2015 Kaggle competition](https://www.kaggle.com/c/diabetic-retinopathy-detection/data). However, is an atypical Kaggle dataset. In most Kaggle competitions, the data has already been cleaned, giving the data scientist very little to preprocess. With this dataset, this isn't the case.
All images are taken of different people, using different cameras, and of different sizes,this data is extremely noisy, and requires multiple preprocessing steps to get all images to a useable format for training a model.

The training data is comprised of 35,126 images, which are augmented during preprocessing.

## Data Pre-processing and data augmentation
In order to make the most of our few training examples, we will "augment" them via a number of random transformations, so that our model would never see twice the exact same picture. This helps prevent overfitting and helps the model generalize better.

In Keras this can be done via the **keras.preprocessing.image.ImageDataGenerator class**. This class allows you to:
* configure random transformations and normalization operations to be done on your image data during training
* instantiate generators of augmented image batches (and their labels) via .flow(data, labels) or .flow_from_directory(directory). These generators can then be used with the Keras model methods that accept data generators as inputs, fit_generator, evaluate_generator and predict_generator.

```Python 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

 ```

![Resulting Image Examples](https://github.com/gregwchase/eyenet/blob/master/images/readme/17_left_horizontal_white.jpg)

## Transfer Learning in Keras
Transfer learning generally refers to a process where a model trained on one problem is used in some way on a second related problem.
Transfer learning has the benefit of decreasing the training time for a neural network model and can result in lower generalization error.
The weights in re-used layers may be used as the starting point for the training process and adapted in response to the new problem. This usage treats transfer learning as a type of weight initialization scheme. This may be useful when the first related problem has a lot more labeled data than the problem of interest and the similarity in the structure of the problem may be useful in both contexts.

### How to use ResNet50 pre-trained model
When loading a given model, the “include_top” argument can be set to False, in which case the fully-connected output layers of the model used to make predictions is not loaded, allowing a new output layer to be added and trained. 
```Python
resnet=ResNet50(input_shape=IMAGE_SIZE +[3],weights='imagenet',include_top=False)

```
A model without a top will output activations from the last convolutional or pooling layer directly. One approach to summarizing these activations for thier use in a classifier or as a feature vector representation of input is to add a global pooling layer, such as a max global pooling or average global pooling. The result is a vector that can be used as a feature descriptor for an input. Keras provides this capability directly via the ‘pooling‘ argument that can be set to ‘avg‘ or ‘max‘.
```python
x = Flatten()(resnet.output)
prediction = Dense(5, activation='softmax')(x)
model = Model(inputs=resnet.input, outputs=prediction)

```
## ResNet50 Model Architecture
Now we’ll talk about the architecture of ResNet50. The architecture of ResNet50 has 4 stages as shown in the diagram below. The network can take the input image having height, width as multiples of 32 and 3 as channel width. For the sake of explanation, we will consider the input size as 224 x 224 x 3. Every ResNet architecture performs the initial convolution and max-pooling using 7×7 and 3×3 kernel sizes respectively. Afterward, Stage 1 of the network starts and it has 3 Residual blocks containing 3 layers each. The size of kernels used to perform the convolution operation in all 3 layers of the block of stage 1 are 64, 64 and 128 respectively. The curved arrows refer to the identity connection. The dashed connected arrow represents that the convolution operation in the Residual Block is performed with stride 2, hence, the size of input will be reduced to half in terms of height and width but the channel width will be doubled. As we progress from one stage to another, the channel width is doubled and the size of the input is reduced to half.

Finally, the network has an Average Pooling layer followed by a fully connected layer having 1000 neurons (ImageNet class output).

![](https://i.stack.imgur.com/gI4zT.png)

## Technologies used
![](https://miro.medium.com/max/600/0*a6XSwHsfvz_oWSSJ.jpg)

## Results
Note that these results are noted after just **10 epochs** because of less compute power availability, this is expected to give much better accuaracy.

|Metric|Value|
|---|---|
|Train(Acc)|60%|
|Valid(Acc)|40%|
