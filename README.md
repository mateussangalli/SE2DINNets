# SE2DINNets
Accompanying code for the paper [DIFFERENTIAL INVARIANTS FOR SE(2)-EQUIVARIANT NETWORKS](https://hal.science/hal-03703287v1/) published at ICIP2022

The layer SE2DIN computes differential invariants plane rotations. Given a function $u$ on the plane and its derivarives $u_x$, $u_y$ $u_{xx}$...
The invariants are given by the expressions[^1]
$$I_{00} = u$$
$$I_{10} = \sqrt{u_x^2 + u_y^2} = \lVert \nabla u \rVert$$
$$I_{20} = \frac{1}{\lVert \nabla u \rVert^2} (u_{xx} u_x^2 + 2 u_x u_y u_{xy} + u_{yy} u_y^2)$$
$$I_{11} = \frac{1}{\lVert \nabla u \rVert^2} (u_x u_y (u_{yy} - u_{xx}) + u_{xy} (u_x^2 - u_y^2))$$
$$I_{02} = \frac{1}{\lVert \nabla u \rVert^2} (u_{yy} u_x^2 - 2 u_x u_y u_{xy} + u_{xx} u_y^2)$$

And their invariant derivatives. In the implementation, invariants after $I_{00}$ are normalized by multiplication with $\lVert \nabla u \rVert$.

[^1]: The expressions for I_20 and I_02 in the paper where missing a factor of 2.

To download and prepare the datasets do:



```
wget http://www.iro.umontreal.ca/~lisa/icml2007data/mnist.zip
wget http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip

unzip mnist.zip -d mnist12k
unzip mnist_rotation_new.zip -d mnist_rot

python prepare_datasets.py
```

Then run train_SE2DINNets.py to train a model based on differential equivariants of SE(2).
