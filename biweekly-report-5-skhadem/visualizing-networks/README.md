# Visualizing Networks
I loved the CAM papers, since they show how the network is actually thinking. I think the results cement the idea that convolutional neural nets are learning useful features, not just learning the general distribution of the data, as many people fear. The cool thing is that you can look at the output of CAM and know if the model is "looking at" the right things. Thus, I wanted to try out the code for CAM, to see if I could get a glimpse at where the network is focusing for different classes.

## Motivation
An explanation I found that explains a good reason why it is important to be able to visualize how a model works:
- Three stages for transparency
    - When the model is worse than humans, transparency is useful to identify how/why the model is failing
    - When the model is on par with humans, transparency is useful to establish trust and confidence in users
    - When the model is better than humans, transparency is useful to teach humans about certain tasks
This is a super cool way to describe why visualizing networks can be key to advancing the field for various stages o fa model's development


- [CAM](./cam/README.md)