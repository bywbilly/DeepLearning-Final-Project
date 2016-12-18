# DeepLearning-Final-Project
Do a toy Tone Classification in Tensorflow, Thenao and Mxnet

#Usage

**data_prosess.py** is used to preprocess the data and it will integrate the training, test and test_new data in two file which is *.f0 and *.engy.

For each file, the format is:
- The number of the features in this training sample, a int
- one float a line, the feature itself
- The label, one number from 1, 2, 3, 4

**Tensorflow Tone.py** is used for training, at the head of this code, you need to change the directory to your own directory.
**tfnnutils.py** is just some implementation for some layers.
