# Testing Understanding of Dimensions by Examininig Popular Architectures
I want to further cement the idea of how the dimensions line up by looking through some of the setups of the popular CNN architectures and make sure I get where the numbers come from.


Some useful formulas: 

For receptive field:

![OW](https://miro.medium.com/max/290/1*V-7WIylrjhDs2VkGYFBvtg.png)
![OH](https://miro.medium.com/max/303/1*4HZOf9zh9F0owsCv-GNqcg.png)

Max pooling size:

![OM](https://miro.medium.com/max/206/1*A3Igi-vAMsOu75yyiVRO1A.png)

- K: number of filters
- Fw,Fh: Filter width/height
- Sw, Sh: stride width/height
- P: padding
- OM: Output matrix
- IM: Input matrix
## LeNet - 5
![Lenet-5](https://miro.medium.com/max/4308/1*1TI1aGBZ4dybR6__DI9dzA.png)

I found [this](https://engmrk.com/lenet-5-a-classic-cnn-architecture/) very helpful to understand the dimensions. Below are the calculations for my own reference. What is a bit confusing, is that the filters have 3 dimensions, with the 3rd dimension being the number of channels in the input (also the number of 2d kernels). These are run on **each** layer, then concacted together.

![First layer](https://engmrk.com/wp-content/uploads/2018/09/LeNet_Layer1.jpg)

The first layer is straightforward, since the input is 2D. 6 5x5 kernels are run on the image, so the output is 28x28x6.


![Second layer](https://engmrk.com/wp-content/uploads/2018/09/LeNet_Layer2.jpg)

The max pooling here serves to reduce the dimensions *per channel*.

![Third layer](https://engmrk.com/wp-content/uploads/2018/09/LeNet_Layer3.jpg)

This is where it gets a little confusing. There are 16 5x5 kernels. Each strides across all 6 input layers, and the result is added together to get one layer. So, for example, kernel_1 sees all 6 channels, the results are added together (with a bias), and the result is layer 1 in the output. The process is repeated for each of the 16 kernels.

![Forth layer](https://engmrk.com/wp-content/uploads/2018/09/LeNet_Layer4.jpg)

The rest of the layers are fully connected, which I understand much better. Looking through these numbers, I feel much more confident in how the dimensions line up. I think my main source of confusion came from the "deepening" of the feature vector, since I did not understand that there were many filters concatenated together.

## AlexNet
The input to AlexNet is a 227x227x3 image, so we are now onto using color images of larger sizes.

![alexnet](https://miro.medium.com/max/700/1*jqKHgwZ8alM3K_JRYO_l4w.png)

![alexnet-2](https://www.learnopencv.com/wp-content/uploads/2018/05/AlexNet-1.png)

What is interesting here is that by using a stride of 1 and a padding of 2in the conv layers (after the first) so the dimension of the width and height of feature vector stays the same, and the depth increases. I believe this is what `padding='same'` does in tensorflow/keras.

The feature vectors get deeper and deeper and then actually get less deep, going from 384 to 256. These values seem so arbitrary, I am curious what work went into picking them.

Another note is that AlexNet uses **overlapping** max pool to reduce the dimensions.

As before, the final fully connected layers take in the flattened feature vector and convert to a softmax-activated 1000 neurons, which correspond to class probability.


## Resources
- [Article](https://medium.com/@RaghavPrabhu/cnn-architectures-lenet-alexnet-vgg-googlenet-and-resnet-7c81c017b848) going through a few different architectures
- [Article](https://engmrk.com/lenet-5-a-classic-cnn-architecture/) on LeNet5 specifically.
- [Article](https://www.learnopencv.com/understanding-alexnet/) on AlexNet specifically