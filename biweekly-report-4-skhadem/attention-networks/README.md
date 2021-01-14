# Attention Networks

## STN
I went back to the work from the STN paper from last report, and finished some visualizations I wanted to create. The last thing I want to do is resizing the digits.

The final product can be found [here](./stn/README.md)

## Various Other Attention Papers

I spend time skimming all of these papers, and included a brief summary here. I spent the majority of my time on [deformable conv nets](./deformable-conv-nets/) and [residual attention networks](./residual-attention-networks/).

- [Convolutional Block Attention Module](https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf)
    - A convolutional block attention module is a simple attention module that infers attention maps along spatial and and channel dimensions. These maps are multiplied to the input feature map for feature refinement. This modules is super lightweight and generalizable, so it can be introduced into any model with very little overhead, since it done end to end using the output labels only.

- [Deformable Convolutional Networks](https://openaccess.thecvf.com/content_ICCV_2017/papers/Dai_Deformable_Convolutional_Networks_ICCV_2017_paper.pdf)
    - Traditional CNNs use a fixed geometric structure in their building blocks (namely, a grid). this paper introduces a _deformable_ convolution and ROI pooling, which are modules that augment their shape based on the feature map provided. They can easily be swapped out for normal conv and max pooling blocks, and can improve the performance of object detection and semantic segmentation.

    - Deeper dive: [deformable-conv-nets](./deformable-conv-nets/)

- [Geometric Deep Learning](https://arxiv.org/pdf/1611.08097.pdf)
    - The scope of this paper is slightly over my head, it is very advanced math. From what I understand, the paper is an overview of different methods to generalize neural networks on non-euclidean spaces, such as high dimensional graphs and manifolds

- [Residual Attention Network for Image Classification](https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_Residual_Attention_Network_CVPR_2017_paper.pdf)
    - This paper introduces stacking attention modules into a network, in order to improve performance on residual networks. These attention modules change adaptively based on the features in a certain layer. Interestingly, the modules contain a feedforward method that combines feedback into a single forward pass. The authors claim that the structure can be scaled up to hundreds of layers.
    
    - Deeper dive: [residual-attention-networks](./residual-attention-networks/)


- [Bilinear CNN Models for Fine-Grained Visual Recognition](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Lin_Bilinear_CNN_Models_ICCV_2015_paper.pdf)

    - This architecture uses two separate CNN streams whose outputs get combined at each location of the image to obtain a single feature vector that is then fed into the classifier head. This allows for translation invariant regions to be used for fine grained image classification

- [Look and Think Twice](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Cao_Look_and_Think_ICCV_2015_paper.pdf)
    -TODO: Basically uses some cool feedback mechanisms, which more closesly resembles how biological vision works