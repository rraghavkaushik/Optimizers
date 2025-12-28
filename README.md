# Optimizers

PyTorch Implementation of Optimizers from scratch.
> **Note:** All implementations are based on the article *“An Overview of Gradient Descent Optimization Algorithms”* by Sebastian Ruder.

## Gradient Descent 

Gradient descent is a way to minimize an objective function J(θ) parameterized by a model’s parameters(θ) by updating the parameters in the opposite direction of the gradient of the objective function ∇θJ(θ) w.r.t. to the parameters. The learning rate η determines the size of the steps we take to reach a (local) minimum. In other words, we follow the direction of the slope of the surface created by the objective function downhill until we reach a valley. (Taken from the article cited below)

## Gradient Descent Variants
There are 3 main variants of gradient descent and they differ in the amount of data used to compute the gradient of the objective function. 

### BGD - Batch Gradient Descent
BGD computes the gradient of the cost function w.r.t. to the parameters θ for the entire training dataset. We perform an update in the direction of the gradients and the learning rate(η) determines how large of an update we perform.

Update rule: 

<img width="261" height="56" alt="image" src="https://github.com/user-attachments/assets/6a4a7f54-8ce6-4fb4-8975-a68cc1b2867d" />

Batch gradient descent will converge to the global minimum for convex error surfaces and to a local minimum for non-convex surfaces.

> It can be very slow and infeasible for datasets that do not fit into the system memory.

### SGD - Stochastic Gradient Descent



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

