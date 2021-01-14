# Biweekly Report 5
## Soroush Khadem

This report, I focused the majority of my time on a paper titles "Learn to Pay Attention." The paper provided some really cool insight and intuition about attention modules, and also had some really interesting results. I decided to spend time reading and understanding the paper, and then running experiments on my own. This was the first time I worked with bigger datasets such as CIFAR-100 and CUB-2011, so I had some frustrations and lessons learned. I am including a list of those here partially for my own reference.

- Always run a sanity check on single epoch
    - Because python is a runtime language, there are many parts of code that can have have bugs, but these only show up when they are run. It is important to at least run through the entire training code once before setting your machine to run for a few hours
    - For example, I had code that saves the model every 10 epochs, but this ended up having a bug. So, after about an hour of training, the code crashed, and I had no saved model to recover
- Tensorboard is key!
    - Being able to glance at live updating plots to ensure nothing weird is happening is key when working with super long training times
    - Especially when running an experiment for a while, it is nice to come back to a few plots to be able to quickly see the initial results
- Use `num_workers=0` when the `dataloader` crashes
    - When starting to work with CUB-2011, there was a crash that kept occuring, and I couldn't get to the bottom of why it was happening because the stack trace was very cryptic
    - This is because the error message from the multithreading of the dataloader does not propogate through. Using no multithreading, then the error becomes clear
    - In my case, it was CUDA being out of memory, which leads to the next item
- Need big hardware for harder tasks.
    - For harder tasks like CUB-2011, the image resolution is key, since some of the features are extremely fine grained. This means that you must use large larger images, whcih take up more space on the GPU
    - I have a 1080 Ti, which has 11 GB of memory. I was trying to train with an image size of 224x224, but this limited me to a batch size of 2, which is almost impossible to learn from.
    - This makes it clear that for some of the hard tasks, it is necessary to use a dedicated server with much higher hardware capability.

Again, I include the list of tasks from last time, as well as some ideas that I didn't have time for which, which I am leaving for next time

- [ ] Cloud Compute
    - [ ] Explore using AWS/Azure/Google Cloud to train networks. Explore pricing, student deals, etc.
    - NOTE: I have been wanting to do this since the 3rd report. Now that I have moved on from MNIST, it is time to actually do this
- [x] Attention Networks [[link](./attention/README.md)]
    - [x] Read through more papers
    - [x] "Learn to Pay Attention" Implementation
    - [x] CIFAR 100
    - [ ] CUB-2011
        - Couldn't get the model to converge. Next time this will be my focus!
- [x] Visualizing Networs [[link](./visualizing-networks/)]
    - [x] Implement CAM
    - [x] Play around with some example images