# JavaCNN (Work in progress)

This project tries to rewrite the [ConvNetJS](https://github.com/karpathy/convnetjs) system built in Javascript.

I've taken the liberty to change some names and I haven't implemented all classes yet but will try to do that in the future.

The net class has the new name of JavaCNN and all the features of creating a network from a definition is removed. Might build those in the future but for this first implementation they aren't required.

This implementation is also made into an video series that you can follow along at [Machine Learning](https://www.youtube.com/playlist?list=PLP2v7zU48xOLt9Hqiu3j3PdBV5cKgqio7)

#### Data
The class Vol in [ConvNetJS](https://github.com/karpathy/convnetjs) it has the new name of DataBlock.

##### DataBlock
Holding all the data handled by the network. So a layer will receive this class and return a similar block as a output that will be used by the next layer in the chain.

##### BackPropResult
When we have done a back propagation of the network we will receive a result of weight adjustments required to learn. This result set will contain the data used by the trainer.

#### Layers
A convolution neural network is built of layers that the data traverses back and forth in order to predict what the network sees in the data.

##### InputLayer
The input layer is a simple layer that will pass the data though and create a window into the full training data set. So for instance if we have an image of size 28x28x1 which means that we have 28 pixels in the x axle and 28 pixels in the y axle and one color (gray scale), then this layer might give you a window of another size example 24x24x1 that is randomly chosen in order to create some distortion into the dataset so the algorithm don't over-fit the training.

##### ConvolutionLayer
This layer uses different filters to find attributes of the data that affects the result. As an example there could be a filter to find horizontal edges in an image.

##### LocalResponseNormalizationLayer
This layer normalize the result from the convolution layer so all weight values are positive, this will help the learning process and shape the result.

##### PoolingLayer
This layer will reduce the dataset by creating a smaller zoomed out version. In essence you take a cluster of pixels take the sum of them and put the result in the reduced position of the new image.

##### FullyConnectedLayer
Neurons in a fully connected layer have full connections to all activations in the previous layer, as seen in regular Neural Networks. Their activations can hence be computed with a matrix multiplication followed by a bias offset.

#### Loss layers

##### SoftMaxLayer
This layer will squash the result of the activations in the fully connected layer and give you a value of 0 to 1 for all output activations.

##### SVMLayer
This layer uses the input area trying to find a line to seperate the correct activation from the incorrect ones.

##### RegressionLayer
Regression layer is used when your output is an area of data. When you don't have a single class that is the correct activation but you try to find a result set near to your training area.

#### Trainers
Trainers take the generated output of activations and gradients in order to modify the weights in the network to make a better prediction the next time the network runs with a data block.

##### AdaDeltaTrainer
Adaptive delta will look at the differences between the expected result and the current result to train the network.

##### AdaGradTrainer
The adaptive gradient trainer will over time sum up the square of the gradient and use it to change the weights.

##### AdamTrainer
Adaptive Moment Estimation is an update to RMSProp optimizer. In this running average of both the gradients and their magnitudes are used.

##### NesterovTrainer
Another extension of gradient descent is due to Yurii Nesterov from 1983,[7] and has been subsequently generalized

##### SGDTrainer
Stochastic gradient descent (often shortened in SGD), also known as incremental gradient descent, is a stochastic approximation of the gradient descent optimization method for minimizing an objective function that is written as a sum of differentiable functions. In other words, SGD tries to find minimums or maximums by iteration.

##### WindowGradTrainer
This is AdaGrad but with a moving window weighted average so the gradient is not accumulated over the entire history of the run. It's also referred to as Idea #1 in Zeiler paper on AdaDelta.
