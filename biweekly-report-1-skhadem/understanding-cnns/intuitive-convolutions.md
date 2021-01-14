## [Intuitively Understanding Convolutions For Deep Learning](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)

This is notes on the above linked blog post.

2D convolutions are simple to understand. Using a sliding "kernel" a 2D convolution essentially transforms one 2D matrix into another. This is essentially a weighted sum, with the value of the weights being the values of the kernel.

Here, this size of the kernel is important since it determines how many features get combined together.

Some important terms:
- Kernel - A matrix that is slid over the input data. Using element-wise multiplication, each patch of pixels that the matrix "sees" corresponds to a single value in the next tensor. The value of the kernel is learned (weights)
- Padding - Many times, the dimensions of the input data does not line up with the size of the kernel, so padding is the process of adding extra 0s to the edges of the data in order for the border to be used.
- Stride - Instead of sliding over every single pixel, by specifying a higher value for the stride, the kernel can move by a value other than 1 over the input. This reduces the dimensions of the next tensor even more. (NOTE: This can be seen as an alternative to using a max pooling layer)
- Filter - When the dimensionality of the input layer goes over 2, a filer is actually a collection of unique kernels
    - For example, in an image (3 channels) a filter could consist of one kernel per channel, so there would be 3 kernels, each getting updated individually

In more dimensions: each kernel slides over the channel, and then the result is summed together to create a single output channel. There is also a bias term that can contribute to this final sum. Furthermore, an arbitrary number of filters can be used together and then concatenated together. *The number of output channels is the number of filters used*.

Ultimately, using convolutions is much more efficient than simply fully connected networks.
- For example, suppose we have a 4x4 input, and want to get to a 2x2 grid. The fully connected way to do it is to flatten the input and map the 16 elements to the 4 outputs, which leads to 16*4 = 64 weights. Instead, using a 3x3 kernel has only 9 weights.

Locality: Due to the nature of kernels, they only see input features from a small local area. In addition, since the same kernel is used to slide across the data, it must be able to generalize. Using backprop means that the kernel must learn just from the local features it produced. **This is why CNNs are mainly used for images**. The direct (geometric) neighbors of the data are used, due to the property of kernels. In images, this makes sense. Pixels near each other are usually related. With other forms of data, this does not work. Consider a simple example like real estate prices in different states. The price of one property does not impact the price of the next property in the list. Thus, CNNs do not apply here since there is no idea of grouped features.

The idea of using filters is common in CV (see something like a Sobel transformation for edge detection). _Filters reduce the dimensions of the data, while making them deeper._ This can be a problem, since as the size gets smaller and smaller, it can become difficult to use small filters to extract filters.

Receptive field: Thinking about stacking convolutions together. Image one layer reducing the size (assume 2D) of the data. The next layer then strides across this smaller image, but now the *receptive field* is bigger in the original image. This is because the same number of pixels in the smaller image represent more pixels in the original image. This idea is what allows the networks to represent more complex features. I.e. the first layer is edges, the next layer is curves, etc. **This is why ther number of filters usually increases**. The lower level features are common to all classes, and the higher level features get more and more specific in what they are looking for. This visualization from the article helps understand this idea:

![Visualization of Layers](https://miro.medium.com/max/257/1*QebJ6hejQh074dYkDkyo6g.png)

Note that the early layers are abstract patterns and textures, but the last layer's channel is looking specifically for bird-like objects. From the article:

> The CNN, with the priors imposed on it, starts by learning very low level feature detectors, and as across the layers as its receptive field is expanded, learns to combine those low-level features into progressively higher level features; not an abstract combination of every single pixel, but rather, a strong visual hierarchy of concepts.

Based on what I learned in Computational Neuroscience, this mimics how the brain works, since there are studies that show the brain uses a *hierarchy* to understand what images it is receiving.

### Conclusion
This blog post was an incredibly useful way to understand how convolutions work at an intuitive level. There were a few really key takeaways that really help my understanding of CNNs:
- By increasing the number of filters per layer, the depth of the feature vector increases
- Distinguishing between "filter" and "kernel" and understanding how the number of channels increase
    - Between layers, the number of channels must match the number of kernels per filter, and the number of filters together leads to more channels
- Stacking layers together provides a hierarchy of features, and the higher level features that come later require more filters since they are more specific

The visualizations provided in the blog were excellent, and I put them here for my own reference:

![Basic 2D conv](https://miro.medium.com/max/535/1*Zx-ZMLKab7VOCQTxdZ1OAw.gif)
![Padding](https://miro.medium.com/max/395/1*1okwhewf5KCtIPaFib4XaA.gif)
![Stride](https://miro.medium.com/max/294/1*BMngs93_rm2_BpJFH2mS0Q.gif)
![Multiple channels](https://miro.medium.com/max/1000/1*8dx6nxpUh2JqvYWPadTwMQ.gif)
![Concact](https://miro.medium.com/max/1000/1*CYB2dyR3EhFs1xNLK8ewiA.gif)


