# Comparison of Various Deep Learning Frameworks
## Tensorflow (Google)
- 1.X: Graphs: Before TF2.0, everything in tensorflow was based on a **graph**. I used this framework for a bit two years ago, and it was confusing. It felt unnatural to do everything within a context manager, and caused the iterations on code to be slower, since things were wrapped up and hidden away. 
- 2.X: The introduction of eager execution made things feel a lot more natural and pythonic.
- Dpeloyment: tensoflow is regarded to be the more production-ready framework. The company I work for uses tensorflow on its cars, and the tools there are pretty built-out. In addition, tensorflow lite provides a way to compress models and optimize for embedded systems, meaning model can run really fast and on memory-limited systems.
- Google colab: Because it is a google product, the popular online compute platform has easy to use built-in tensorflow working. 
- Widely supported: There is language support in java, swift, C++, etc. so native code can be written for almost any platform.


## PyTorch (Facebook)
- Pytorch was released after tensorflow, but quickly gained popularity as it was more natural to use within python. It is typically used for research, but is growing in popularity within production. Pytorch is based on torch, which is a framework that does fast computations in C. Thus, it has improved memory and optimization.
- Pytorch offers very fine-grained control over all aspects of dep learning. By subclassing models or layers, custom implementations can be written easily, and designs can be expanded and iterated easily.
- Using datasets allows for much more contained modules of code that can be re-used easily

## Keras
High-level wrapper of tensorflow, but seems to be coupled in, as tensorflow has `tf.backend.keras`. Keras makes it easy to get up and running, as models can be trained in very few lines of code. I am not a huge fan of keras, since it hides how things are working, and doesn't allow for fine-tuned control. I like writing out the training loops to see what is happening, instead of simply calling `model.fit()`.

## Caffe (Berkley)
Some [slides](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.gc2fcdcce7_216_0) that introduce the framework. Did not have time to really dig into what it offers, and have also not seen this as a widespread framework.

## In the end...
- Both pytorch and tf have built-in dataset loading helpers, which are very nice
- With TF2.0, the APIs are starting to feel very similar
- Both have tools like tensorboard integrated
- Both have widespread support in the deep learning community 
- Being familiar with both is necessary
- However, I believe that favoring one will allow for more mastery, and the ability to get past the nuances of a framework and into the actual deep learning concepts
- This is very subjective, but, the code I have seen in pytorch feels much *cleaner* than the tensorflow code. This is most likely due to my C++ background and object-oriented programming style.

Therefore, I want to mostly use pytorch, but also continue to do a few examples in tensorflow here and there.


## Resources
- https://medium.com/coding-blocks/eager-execution-in-tensorflow-a-more-pythonic-way-of-building-models-e461810618c8
- https://realpython.com/pytorch-vs-tensorflow/
- https://towardsdatascience.com/pytorch-vs-tensorflow-spotting-the-difference-25c75777377b