# Notes on Forward Forward Algorithm
The forward-forward algorithm is a novel method for training neural networks as an alternative to backpropagation.

It is based on the idea of replacing the forward and backward passes of backpropagation by two forward passes, one with positive and one with negative weights. This approach has two significant advantages over backpropagation:

* **It is more efficient,** since it does not require the computation of gradients.
* **It is more robust to local minima,** since it does not rely on the smoothness of the loss function.

However, the forward-forward algorithm is somewhat slower than backpropagation and does not generalize as well to new data. Overall, it is a promising alternative to backpropagation, but further research is needed to determine its full potential.

**Here are some additional insights from the sources:**

* The forward-forward algorithm works well in relatively small networks, but it does not scale well to large networks.
* The forward-forward algorithm is more efficient than backpropagation, but it is also more computationally expensive.
* The forward-forward algorithm is more robust to local minima than backpropagation, but it is also more likely to overfit to the training data.
* The forward-forward algorithm can be used to train neural networks for a variety of tasks, including image classification, natural language processing, and speech recognition.

Overall, the forward-forward algorithm is a promising alternative to backpropagation, but it is still under development and there are some limitations to its performance. Further research is needed to determine its full potential.

Having seen it's promising results, and lack of conclusive evidence that it scales better on larger datasets and the kindof architecture that's best, we've developed a library which helps easily scalable Forward Forward network which is compatible with other PyTorch components.
Code: [github source](https://github.com/shashvatshah9/FFPytorch)
Project summary: [project summary](https://github.com/shashvatshah9/shashvatshah9.github.io/blob/master/assets/dl_final_project_report.pdf)

References:
1. https://github.com/mohammadpz/pytorch_forward_forward
2. Geoffrey Hinton's talk at NeurIPS 2022.
