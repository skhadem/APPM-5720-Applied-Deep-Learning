# Biweekly Report 2
## Soroush Khadem

This report, I focused the majority of my time implementing ResNet from scratch, which I would call exploitation. The ideas behind residual connections seem so important, that I thought I should dedicate a good amount of time to it. When doing so, I found myself spending a lot of time using PyTorch, reading documentation, and understanding every line of code I wrote. Then, when I finished writing it, it took some time to debug it, since I had made a mistake in the dimensions between bottleneck layers (see the description at the end of the [notebook](./resnet/resnet.ipynb)). After getting the model to work, I spent a lot of time training it in various ways, tweaking hyperparams, visualizing results using tensorboard, etc. There is an addicting feeling to watching live loss graphs, and rooting for your model to go lower! I found myself watching the graphs for minutes at a time.

I also spent a bit of time (maybe 20%) on exploration, learning more about the basics of GANs and exploring projects in that space. 

Tasks:

- [ ] Adversarial Learning
    - [ ] (From last report) More rigorous coding examples - maybe try to show some confusion matrices
    - [x] (From last report) Begin to explore GANs
- [x] Compare Frameworks
    - [x] (From last report) Do some research into the pros/cons of each
    - [x] Simple MNIST with PyTorch
- [x] ResNet
    - [x] Re-create ResNet arch. in code - compare dimensions to make sure matches
    - [x] Run on CIFAR-10
    - [x] See if performance is the same
    - [x] Run multiple depths and compare
- [ ] Dense Conv Net
- [ ] Style transfer
    - [ ] Start to learn about the basics

Similar to last report, I have left the unchecked items as a starting point for the next one. 

Some ideas I had for next time:
- Adversarial Learning
    - Try out a simple MNIST example
- ResNet
    - Tweak existing model to try and match paper's performance
- Cloud Compute
    - Get an introduction to using AWS/Azure/Google Cloud to train networks. Explore pricing, student deals, etc. Although my 1080 Ti is pretty nice, when in comes to GANs I will need some more compute.

Note: I am liking my strategy so far of keeping the categories similar across reports, it feels like one continuos research project. In addition, I am getting better at coming up with a reasonable plan for tasks given the time frame. The loss function of my incomplete tasks is decreasing :grin: