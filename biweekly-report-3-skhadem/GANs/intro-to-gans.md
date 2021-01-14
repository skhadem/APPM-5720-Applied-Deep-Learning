# Introduction to GANs from NIPS 2016, Ian Goodfellow
Talk: [link](https://www.youtube.com/watch?v=9JpdAg6uMXs&ab_channel=PreserveKnowledge)

## Notes
- The term is in flux, can be used to describe both old and new ideas
- At first: simply using adversarial examples to train on, i.e. using worst case examples (see biweekly report 1)
- In GANs, both players are neural networks - output of generative model is fed into the other model
- One of the players is always trained to do as well as possible on the worst case examples (Discriminator)
    - Denoted as function D, with output from [0,1] signifying [fake, real]
- Other one is generating the worst case examples
    - Function G that takes in noise (z) and outputs a sample x 
- The output of generator is fed into the discriminator: D(G(z))
    - D wants to make D(G(z)) close to 0
    - G wants to make D(G(x)) close to 1
- Applying some game theory, and looking in the long-term, if both networks work perfectly, we would expect the generator to produce perfect examples, so the discriminator would be left to output 0.5 for each example -> gives rise to minimax loss
- Minimax
    - Equilibrium is a saddle point of the discriminator loss
    - Generator minimizes log probability of the discriminator being correct
- Can actually solve for the optimal D, ends up being a ratio of generated data and real data. Using this, can calculate different divergences
- So, the big picture, the goal is to use supervised learning to estimate the ratio, thus yielding a perfect discriminator
- Can do arithmetic in vector space (See Radford, et. al.)
    - ex: Man with glasses - Man + Woman = Woman with Glasses
     - Looks cool!
- Non convergence
    - Optimization can lead to local minimums
    - There may not even be a equilibrium
    - Now, with two optimizations at the same time, much harder

## Seen up until: 16:43