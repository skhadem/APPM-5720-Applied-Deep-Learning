# Weakly Supervised Object Detection (WSOD)
The field of WSOD involves training models to detect objects in images by only using the high level image label. This work is newer in the field, and it has huge advantages since annotating images with bounding boxes is much more human labor intensive then simply assigning a high level label to an image. I have naturally been drawn to this in one of my last reports: when I worked with the spatial transformer network (STN), I was most fascinated by the visualizations where the bounding box could be drawn around an MNIST digit. The idea of localization within the image was inherent to the model. Using attention is thus a natural approach to solving WSOD.

Before diving in to some of the papers that use attention to perform WSOD, I wanted to start with the basics.

First, I wanted to get familiar working with PASCAL VOC, visualizing bounding boxes, IOU, etc.
    
- [Exploratory notebook](./explore_voc.ipynb)

Then, I worked through a paper I saw get cited many times, and one of the first papers to use deep networks for WSOD, simply titled "Weakly Supervised Deep Detection Networks".

- [Summary and experiments](./wsddn/)

I also began reading some attention-based WSOD methods:

- [Summaries](./attention-wsod.md)