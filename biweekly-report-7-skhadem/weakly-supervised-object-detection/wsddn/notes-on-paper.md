# [Weakly Supervised Deep Detection Netwokrs (WSDDN)](https://arxiv.org/pdf/1511.02853.pdf)

## Introduction
At the time of publication (2016), the authors mention that there aren't many solutions to weakly supervised object detection. This work addresses this deficit by using a deep neural network to simultaneously perform region selection and classification. The core idea is that the network is trained simply as a classifier, and it implicity learns detectors that are better than other solutions. In addition, the model outperforms the SOA in other classification tasks.

CNNs have the ability to learn geometric relationships between features that generalize well between tasks. In particular, CNNs pre-trained on ImageNet generalize extremely well. The hypothesis is that because pre-trained CNNs can be applied to other tasks, they should contain meaningful representations of data. (This is a common thought about CNNs as feature extractors). Becuase CNNs are mostly invariant to locations of objects within an image, they may already implicty learn the location of objects.

The idea is that given an existing network trained on ImageNet, this architecture expands it to reason explicitly about multiple image regions $R$. Given an image, the first step is to extract region-level descriptors, by using spatial pyramid pooling layers after the conv layers. Then, two different data streams are used: one associates a class score to each region (recognition), and the other represents the idea that a single region is the best to use (detection). The two streams are combined to predict a class for the image as a whole.

The most common technique is multiple instance learning (MIL). MIL alternates selecting regions in the image, and estimating the appearance of the model in the selected regions. Thus, MIL uses the appearance to select regions, whereas this work uses a parallel detection branch, such that there is no cross-talk between the two tasks. The authors claim that this is why their method avoids getting stuck in local optima, like most weakly supervised detection tasks. This design also allows for little fine-tuning by hand. Thus, it is as effecient as R-CNN, since it only needs back prop.

## Method
Three key ideas to the architecture:
1. Use pre trained CNN on ImageNet
2. Modify the CNN to work with WSDDN
3. Train/fine tune WSDDN on a target dataset (only using image labels)

In order to modify the CNN: replace the last pooling layer (after the ReLU in last conv) with a spatial pyramid pooling layer. This essentially takes in an image and region and produces a feature representation that combines global and local information. REgions are supplied from a region proposal mechanism, namely wither Edge Boxes or Selective Search Windows. Modify the spatial pooling pyramid to take in a list of regions, so the output is a concatonation of all the features. Then, the features are processed by two fully connected layers with ReLU activation. The last layer then branches into the two data streams:
    - The classification stream is simply another linear layer that maps to the output number of classes, with a softmax activation over the classes
    - The detection stream scores regions using a different linear layer, with a softmax over the regions
Note that the output of both data streams is a C x |R| matrix, where C is the # of classes and |R| is the number of regions. Almost identical operations are done, but the difference lies in the softmax. Put very nicely:

```
Hence, the first branch predicts which class to associate to
a region, whereas the second branch selects which regions
are more likely to contain an informative image fragment
```

The score for each region is an element-wise product of the outputs of the softmaxes. _Non-maxima_ suppression is used, which is a method to remove regions with IOU over 0.4 with already selected regions.

The score for each class is the sum over al regions for the class index. Note that images are allowed to contain multiple classes, while regions should only have a single class.

For training, the labels are converted to the format $y_i = {-1, 1}^C$. The loss function is:

$ E(w) = \frac{lambda}{2}||w||^2 + \sum_{i=1}^n\sum_{k=1}^C\text{log}(y_{ki}(f(x_i | w) - \frac{1}{2}) + \frac{1}{2}) $

Where $n$ is the number of images and $f$ is the model. This can be seen as a sum of $C$ (number classes) binary log-loss terms.
