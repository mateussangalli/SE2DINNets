# SE2DINNets
Accompanying code for the paper "DIFFERENTIAL INVARIANTS NETWORKS FOR SE(2)-EQUIVARIANCE" to be submitted at ICIP2022

To download and prepare the datasets do:



```
wget http://www.iro.umontreal.ca/~lisa/icml2007data/mnist.zip
wget http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip

unzip mnist.zip
unzip mnist_rotation_new.zip

python prepare_datasets.py
```

Then run train_SE2DINNets.py to train a model based on differential equivariants of SE(2)
