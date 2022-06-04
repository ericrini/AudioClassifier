# Transfer Learning
This is an example of using [ML.NET](https://dotnet.microsoft.com/en-us/apps/machinelearning-ai/ml-dotnet) to train a transfer learning model based on [ResNet](https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035). It uses a set of 2311 female samples and 3682 male samples from the [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) data set for training and achieves at least 86% accuracy. The training is done with a .NET and C++ implementation that runs in a single process and does not emulate any Python runtime.

## Native Dependencies
* [tensorflow](https://www.tensorflow.org/install/pip)
* [cuda-toolkit](https://developer.nvidia.com/cuda-toolkit)
* [cuDNN](https://developer.nvidia.com/rdp/cudnn-download)