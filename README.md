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
