# Biweekly Report 3
## Soroush Khadem

For this report, the majority of my time was spend in the spatial transformer networks section. After learning about them in class, this network is so fascinating I felt like I needed to get my hands dirty trying it out. I also spent a lot of time tuning my hyperparameters from the ResNet implementation last time. I spent the minority of the time learning a little bit about style transfer and adversarial learning.

Tasks:

- [x] ResNet [[link]](./resnet/README.md)
    - [x] Tweak existing model to try and match paper's performance
- [ ] Spatial transformers [[link]](./spatial-transformer-networks/README.md)
    - [x] Take more detailed notes, starting from the in class ones
    - [x] Implement it
    - [ ] Recreate gifs/videos here](https://drive.google.com/file/d/0B1nQa_sA3W2iN3RQLXVFRkNXN0k/view)
- [ ] Style transfer [[link]](./neural-style-transfer/README.md)
    - [x] Start to learn about the basics
    - [ ] Take in depth notes on a paper
    - [x] Try and download some pre-trained models to just evaluate
- [ ] Adversarial Learning [[link]](./GANs/README.md)
    - [x] Watch some video introductions
    - [ ] Try out a simple MNIST example
- [ ] Cloud Compute
    - [ ] Explore using AWS/Azure/Google Cloud to train networks. Explore pricing, student deals, etc.

Similar to last report, I have left the unchecked items as a starting point for the next one. 

Some ideas I have for next time:
- Spatial transformers
    - I really want to figure out how to extract the bounding box from the inverse affine transform. I think I need to break out the linear algebra and work through the math
    - Try and figure out why the rotation wasn't working -> maybe explore using a different transform

Cool papers I found for my own reference for later:

- [Image to Image Cycle GAN](https://arxiv.org/pdf/1703.10593v7.pdf)
- 