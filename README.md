# PSO
This repoistory contains research about Particle Swarm Optimization (PSO) and it's implementation to optimize Artificial Neural Network (ANN)

## Including
- [The Persentation of PSO](https://github.com/aboelkassem/PSO/blob/master/paper/PSO.pptx)
- [The Paper of PSO](https://github.com/aboelkassem/PSO/blob/master/paper/PSO%20Paper.pdf)
- Code for PSO is from: [kuhess/pso-ann](https://github.com/kuhess/pso-ann)
- Used [MNIST](https://en.wikipedia.org/wiki/MNIST_database) Dataset for Training ANN using PSO which you can download it from
  - [The MNIST training-set](https://www.python-course.eu/data/mnist/mnist_train.csv)
  - [The MNIST test-set](https://www.python-course.eu/data/mnist/mnist_test.csv)
  
 ## Why Traning Neural Network with Particle Swarm Optimization instead of Gradient Descent
 - Motivation
   - Gradient Descent requires differentiable activation function to calculate derivates making it slower than feedforward
   - To speed up backprop lot of memory is required to store activations
   - Backpropagation is strongly dependent on weights and biases initialization. A bad choice can lead to stagnation at local minima and so a suboptimal solution is found.
   - Backpropagation cannot make full use of parallelization due to its sequential nature

 - Advantages of PSO
   - PSO does not require a differentiable or continous function
   - PSO leads to better convergence and is less likely to get stuck at local minima
   - Faster on GPU
  
## Environment
 - Windows 10
 - AMD GPU radeon 530
 - Python 3.9
 - matplotlib	3.3
 - numpy	1.19.5
 - scikit-learn	0.24
 - scipy	1.6.0	
 
 ## Run example with MNIST
 ```bash
 python example_mnist.py
 ```

## Demo and The Efficient of Results
the following diagram and reports shows the performance of testing data of the dataset including 10 classes (digits classes)

<img src="https://res.cloudinary.com/dvdcninhs/image/upload/v1610484745/testingPSO_eejk9w.png" width="300" hight="300"/>
<img src="https://res.cloudinary.com/dvdcninhs/image/upload/v1610484745/plotPSO_oiqreo.png" width="300" hight="300"/>
