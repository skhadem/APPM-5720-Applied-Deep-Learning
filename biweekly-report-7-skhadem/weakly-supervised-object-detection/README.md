# Weakly Supervised Object Detection (WSOD)

I spent some time reading papers and taking notes on the bigger pictures and ideas in them, in order to get a sense of the space and how to try and come up with creative ideas on something to try for my final project. Most likely, I will take ideas from multiple papers and try combining them to see the results. The majority of my notes are on paper, but the main ideas are here for reference.

# Improving WSDDN
## [Weakly Supervised Cascaded Convolutional Netowrks](https://arxiv.org/pdf/1611.08258.pdf)
This paper produces a method to get the region proposals from the network itself, by training a cascade of models, one which produces activations maps, and another that performs classification on some extracted features from the activation map. There is also a variation that uses three cascading models in order to also produce a segmentation map, but these results are not explored much.

## [Online Instance Classifier Refinement (OICR)](https:/arxiv.org/pdf/1704.00138.pdf)
Detailed summary in last report: https://github.com/CU-Boulder-APPM-X720-Fall-2020/biweekly-report-6-skhadem/blob/master/weakly-supervised-object-detection/attention-wsod.md

This network proposes a way to add new positive samples by iteratively finding overlapping proposals, and combining them in a learned way. This helps cover the whole image instead of just one part.

# Using Clustering
There were a few papers I saw that use some sort of clustering to add information to the latent variable (the ROI inputs).
## [Weakly Supervised Object Detection with Convex Clustering
](https://openaccess.thecvf.com/content_cvpr_2015/papers/Bilen_Weakly_Supervised_Object_2015_CVPR_paper.pdf)
This paper introduces a "soft" similarity between each propoesed region and and a set of "exemplars", which is learned during training. To do this, a latent space support vector machine is used to generate the regions, which enforces similarity between the region and other positive examples of that class.

## [Weakly Supervised Object Localization with Latent Category Learning](http://vigir.missouri.edu/~gdesouza/Research/Conference_CDs/ECCV_2014/papers/8694/86940431.pdf)
This paper makes the important note that although it is important to learn a "tight" detection around an object, there can be important semantic information in the background, so simply throwing that out would lose information. Their method learns latent categories related to a class like "sky" or "cloud" for "aeroplane", and then can use this information in a weakly supervised fashion to use all the relevant information, while still separating the object from the background.a

# Using Attention
## [CASD](https://arxiv.org/pdf/2010.12023v1.pdf)
A detailed summary is in the last report: https://github.com/CU-Boulder-APPM-X720-Fall-2020/biweekly-report-6-skhadem/blob/master/weakly-supervised-object-detection/attention-wsod.md

The biggest takeaway is that the network uses augmentation as a part of the training process explicitly. The same region is transformed multiple times, passed through the attention module, and the outputs are compared for consistency.

This paper is still the most interesting of the ones I have read, and I definitely plan on trying it out next. 

# Re-localization
## [Deep Self-Taught Learning for Weakly Supervised Object Localization](https://arxiv.org/pdf/1704.05188.pdf)
One main difference in most of the state of the art and the work I did during this report is that there is a re-localization step that I did not do. This seems to make the most sense to me, as there needs to be some feedback from the model to the bounding boxes. The main idea behind this paper is to let the detector learn low level features for "tight" positive samples, and then use that information to re-train the entire network from scratch. "Tight" is defined based on a dense subgraph for each region proposal.

This paper is complicated but has some really interesting ideas that seem to be SOA.

# Resources
- https://hbilen.github.io/wsl-cvpr18.github.io/assets/wsod.pdf
- https://medium.com/visionwizard/weakly-supervised-object-detection-a-precise-end-to-end-approach-ed48d51128fc
