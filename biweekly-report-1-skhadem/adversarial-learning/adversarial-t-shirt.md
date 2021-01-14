# Adversarial T Shirt
I want to explore this [paper](https://arxiv.org/pdf/1910.11099.pdf) that describes a method of creating an adversarial t-shirt against YOLO, one of the most popular object detection frameworks. Ultimately, after skimming the paper, I think I should wait until after we have covered object detection in class so that I have a better understanding.

From an initial reading, the paper is an improvement of an original method (pictured below) that is much more robust since it uses physical information from a t shirt to improve the pattern. The authors explore various different poses, something that was not done in the first paper.

Some cool images showing the adversarial patch in action.

![orignal](https://miro.medium.com/max/399/1*KJxhSpVQI5cxboupF2XU9Q.png) 
![the patch](https://gitlab.com/uploads/-/system/project/avatar/11900787/object_score.png)

![better](https://miro.medium.com/max/700/1*XuFqv8nJYLeWFBM_2Gz7SQ.png)

# Resources
- [Article](https://towardsdatascience.com/avoiding-detection-with-adversarial-t-shirts-bb620df2f7e6) that helps explain the papers.
- [Paper](https://arxiv.org/pdf/1910.11099.pdf)- newer one that uses the projection of the shirt
- [Original Paper](https://arxiv.org/pdf/1904.08653.pdf) that simply uses a patch held by a person
- [More on adversarial examples for object detection](https://arxiv.org/pdf/1707.07397.pdf)
- [Video example](https://twitter.com/hardmaru/status/1120255078219149312) of the patch paper