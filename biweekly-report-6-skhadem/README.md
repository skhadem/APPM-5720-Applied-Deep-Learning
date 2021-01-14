# Biweekly Report 6
## Soroush Khadem

This week, my time was split into two main areas. I furthered my experiments with the attention net I used last time (Learn to Pay Attention) in an attempt to work with the CUB-2011 dataset for fine-grain classification of bird species. This dataset is extremely challenging, and it is exciting to try and get good results on it. I ran a few experiments, mostly experimenting with which layers from my training on CIFAR-100 I transferred to the new net. This ranged from all to all but the attention layers to all but the classification layers. In addition, I tried "freezing" vs. not "freezing" the layers (i.e. allowing to train). These experiments were time consuming since the training takes a few hours. However, I really enjoyed the process: I liked watching the loss and cheering for my model.

The other area I focused on is weakly supervised object detection (WSOD). This topic involves trying to learn bounding boxes in an image using only the image-level label. This is a very challenging task, but us really important since it could allow for data to be labelled much faster. Naturally, since I have been focusing on attention networks, I was drawn to the approaches that use this strategy. I read and summarized a few papers that use this approach, but ended up spending the majority of my time familiarizing myself with some of the more foundational approaches in the space, in addition to simply getting comfortable with object detection basics. I played around with IOU, visualizing the VOC dataset, and using a region proposal strategy called Edge Boxes.

I spent a minority of the time looking into cloud platforms, but quickly found that Google Colab is the most promising, since the other platforms (AWS, Azure) are less intuitive and require more work to set up. Thus, I did not investigate much further.

- [CUB-2011 with Attention Net](./attention/)
- [WSOD](./weakly-supervised-object-detection/)
- [Brief Summary of Using Cloud platforms](./cloud-platforms/)


## Final Project Idea
After spending a lot of time on WSOD, especially with attention, I am thinking that this area seems promising for a final project. The task is really difficult, and there are not a lot of easily accessible code examples to work from. Thus, it seemse like a good final project would be recreating some of the papers, comparing results, and then maybe trying out some new ideas on top of a previous paper (not sure what this would like yet). One initial idea is to train a supervised object detector on one dataset, and then distill the box regression part onto a different dataset with a weakly supervised approach. What do you think?