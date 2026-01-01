# Optimizers

PyTorch Implementation of Optimizers from scratch.
> **Note:** All implementations are based on the article *“An Overview of Gradient Descent Optimization Algorithms”* by Sebastian Ruder.

## Gradient Descent 

Gradient descent is a way to minimize an objective function J(θ) parameterized by a model’s parameters(θ) by updating the parameters in the opposite direction of the gradient of the objective function ∇θJ(θ) w.r.t. to the parameters. The learning rate η determines the size of the steps we take to reach a (local) minimum. In other words, we follow the direction of the slope of the surface created by the objective function downhill until we reach a valley. (Taken from the article cited below).

A simple GIF demonstrating gradient descent on a convex parabolic loss surface, where iterative updates guide the parameters toward the global minimum is given below:
<p align="center">
  <img src="https://doimages.nyc3.cdn.digitaloceanspaces.com/010AI-ML/content/images/2018/05/68747470733a2f2f707669676965722e6769746875622e696f2f6d656469612f696d672f70617274312f6772616469656e745f64657363656e742e676966.gif" width="420"/>
  <br>
  <em>Gradient descent on a convex loss surface. The ball rolls downhill and converges to the global minimum.</em>
</p>



## Gradient Descent Variants
There are 3 main variants of gradient descent and they differ in the amount of data used to compute the gradient of the objective function. 

### i. BGD - Batch Gradient Descent
BGD computes the gradient of the cost function w.r.t. to the parameters θ for the entire training dataset. We perform an update in the direction opposite to the gradient of the loss function and the learning rate(η) determines how large of an update we perform.

Update rule: 

<img width="261" height="56" alt="image" src="https://github.com/user-attachments/assets/6a4a7f54-8ce6-4fb4-8975-a68cc1b2867d" />

Batch gradient descent will converge to the global minimum for convex error surfaces and to a local minimum for non-convex surfaces.

> It can be very slow and infeasible for datasets that do not fit into the system memory (as the entire dataset must be available to compute gradients).

### ii. SGD - Stochastic Gradient Descent

<img width="354" height="49" alt="image" src="https://github.com/user-attachments/assets/7c7790be-fe87-4074-acb2-e2759ff96f1f" />

There are two problems associated with batch gradient descent, one being that it has strong convergence guarantees for convex objectives, but in non-convex settings it can get stuck at saddle points or poor local minima due to the complex geometry of the loss landscape. The other one being that, it performs redundant computations for similar datapoints in a large dataset for every parameter update. 

These limitations motivate stochastic gradient descent (SGD), which mitigates these issues by using stochastic estimates of the gradient. In SGD, parameters are updated using individual training samples, resulting in faster updates, improved scalability, and support for online learning.

Does using a random vector from the training sample always help move towards convergence? The answer is mostly yes. This is because the SGD gradient is unbiased.

**Why is the SGD gradient unbiased?**

Let the stochastic gradient for a randomly sampled data point (i) be

$$
g_i(\theta) = \nabla_\theta J(\theta; x^{(i)}, y^{(i)})
$$

where the index (i) is sampled uniformly from ({1,...,N}).

Taking expectation over the random choice of (i)

$$
\begin{aligned}
\mathbb{E}*i \left[ g_i(\theta) \right]
&= \mathbb{E}*i \left[
\nabla*\theta J(\theta; x^{(i)}, y^{(i)})
\right]  &= \sum*{i=1}^{N} \frac{1}{N}
\nabla_\theta J(\theta; x^{(i)}, y^{(i)}) 
&= \nabla_\theta J(\theta)
\end{aligned}
$$

So, on average, SGD points in the exact same direction as batch gradient descent, even though each individual update uses only a single data point.


### iii. Mini-Batch Gradient Descent

All implementations are based on the paper 'An overview of gradient descent optimization algorithms' by Sebastian Ruder.

## References

```bibtex
@article{ruder2016overview,
  title={An overview of gradient descent optimization algorithms},
  author={Ruder, Sebastian},
  journal={arXiv preprint arXiv:1609.04747},
  year={2016},
  url={https://arxiv.org/abs/1609.04747}
}
```

