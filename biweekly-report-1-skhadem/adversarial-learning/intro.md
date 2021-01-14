# Adversarial Examples

### What is an Adversarial Example?
![An Adversarial Example](./graphics/panda_example.png)

An adversarial example is like an optical illusion to machine learning model. It is a way of tricking the model into making an incorrect decision. The examples are created by adding noise in a very intentional way. [This](https://arxiv.org/pdf/1412.6572.pdf) arXiv paper discusses why adversarial examples fool networks. An example from the paper is clarifying: input data that is classified as a "panda" by GoogLeNet is perturbed by adding a low amount of noise in a specific way (by aligning with the weights very closely), and the model now predicts the label "gibben". The amount of noise is so low that as humans we can clearly tell that the second image is still that of a panda, which is not always the case. What is fascinating is that sometimes the same examples are misclassified by a wide variety of the top models, showing that maybe adversarial examples cab be a clue into blind spot of neural networks. I am very interested to explore work that has been done to use adversarial examples to improve the performance of neural nets (such as using GANs to create more data).

Some studies have suggested that adversarial examples arise due to the extreme nonlinearity of deep networks. But, using more linear models makes for more efficient training. Thus, the fundamental trade-off is designing models that generalize well vs models that are easier to train. Another reason could perhaps be the failure of current regularization methods. The paper attempts to show using adversarial learning (using adversarial methods while training) can provide a better regularization method than current methods such as dropout.

Some interesting notes on previous work:
- Box-constrained L-BFGS can reliably find adversarial examples
    - Limited memory Broyden-Fletcher-Goldfarb-Shanno algorithm is an iterative method for solving unconstrained nonlinear optimization instances. This method is a Quasi-Newton method, since it approximates the Hessian. [source](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm)
- In some cases, the adversarial example is so close to the original image that it is indistinguishable to the human eye
- The same example fools many different networks with different architectures
- Training on these examples can help, but are oftentimes not practical due to the expensive nature of the optimization

### How Adversarial Examples Work
Note that the precision of an individual input feature is limited. For example, pixels are oftentimes in the range 0-255, so any information discrepancy below 1/255th is discarded. Therefore it is not rational for the classifier to respond differently to an input *x* versus *x + eps* for an epsilon less than the precision of the input feature. However, considering the dot product between a weight vector and the perturbed input, one can see that the product of the weight and the noise can grow large enough to cause the input to fall into a different category. We can maximize the effect of the noise by choosing the value of *eps* based on the weights.

Another way to look at this is by recalling that what weights essentially do is assign higher value to certain features that align with the weights. Thus, setting up the noise to align closely with the weights will force the network to pay attention to noise, even in the case where the magnitude of other signals is much higher. The way to align with the weights is by creating a noise vector that is simply the sign of the gradient of the loss function, times some epsilon that is low enough to allow the noise to "slip" below the threshold of the precision of a single input feature. (In this case, a pixel). This method is known as the "fast gradient sign method", since the gradient is already found using backpropagation anyway.

By cranking up the value of *eps*, it is possible to get an error rate of **99.9%** on the MNIST test set. What is even more worrisome is that the average confidence is 79.3%, meaning that not only are the models incorrect, they are confident in their wrong estimates. Note though that the noise is so high that the inputs become unrecognizable by the human eye, unlike the panda/gibbon example. The paper includes a figure that shows the perturbation on 3s/7s.

[*Question*: How does one pick the two classes to mix up? It is unclear if this choice affects the optimal noise choice, and if so, how it is done. TODO: Recreate some of the results in code]

## Dangers of Adversarial Attacks
It is clear that adversarial examples can be useful to augment training sets in order to allow networks to better generalize. But, due to their nature, they can also be used to maliciously attack a system that depends on neural networks, and the effects can be catastrophic. For example,
- An adversarial image could be placed over a stop sign, going undetected by people but causing self driving cars to misclassify it
- Similarly, adversarial paint could be applied to cars to "hide" them from self driving cars
- License plates could get the same treatment, allowing cars to go undetected on toll lanes
- Reinforcement agents can also be manipulated by adversarial attacks, causing some efforts such as WoW agents to get thrown off during training

## Attempted Defenses
With the current technology, providing more robust generalization (such as better dropout and weight decay) is not a sustainable way to prevent adversarial attacks. Currently, the ways to protect a model from such attacks are:
- Adversarial Training: As mentioned above, using adversarial examples in the training set allows the model to avoid getting fooled
- Defensive Distillation: A separate model can be used to output confidence of a class, so that the learning is "distilled" between multiple models. This is more robust to adversarial tweaks, since the second model is trained on hard class labels, making its surface smoothed in the directions that the weights are strong in.

"Gradient masking" was a method that failed to protect against adversarial attacks. The idea here is to deny the attacker access to a useful gradient. This idea seems to make sense, since the construction of an adversarial attack uses the gradient of the model. In essence, an attacker could take an image of an airplane, test which direction in image space gets the probability of a cat to increase, and they perturb the image towards this direction. The first thought is to hide the output probabilities. So, instead of outputting probabilities, we could output just the class. However, this added step is pretty simple to break, as an attacker could simply train their own model using the outputs of the target model, and eventually get to the same point.

![Attempted Defense](https://openai.com/content/images/2017/02/adversarial_img_3.png)

# Resources
- [Attacking Machine Learning with Adversarial Examples](https://openai.com/blog/adversarial-example-research/)
    - OpenAI is an organization, founded by Elon Musk, that is one of the leading AI groups, behind the incredibly novel GPT language models. This article is written as an introduction to what adversarial attacks are, in the context of machine learning. The article also goes into why adversarial examples can be so hard to secure systems against.
- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/pdf/1412.6572.pdf)
    - Written by Ian Goodfellow, one of the deep learning greats. Good introductory article, explains in more technical detail how adversarial examples work. Also provides some intuitive understanding as to how they can be used to improve training.
- [Cleverhans](https://github.com/tensorflow/cleverhans)
    - A product of Ian Goodfellow, a package that is aimed at using adversarial examples to evaluate the performance of neural nets. Seems to still be in progress, but has some cool examples.