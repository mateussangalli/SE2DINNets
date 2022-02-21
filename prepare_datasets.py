import numpy as np
import matplotlib.pyplot as plt
import os

out_dir = ''
mnist12k_dir = 'mnist12k'
mnist_rot_dir = 'mnist_rot'

data_train_val = np.loadtxt(os.path.join(mnist12k_dir, 'mnist_train.amat'))
data_test = np.loadtxt(os.path.join(mnist12k_dir, 'mnist_test.amat'))unzip mnist.zip
unzip mnist_rotation_new.zip

images_train_val = data_train_val[:, :-1].reshape(-1,28,28,1)
labels_train_val = data_train_val[:, -1]

images_test = data_test[:, :-1].reshape(-1,28,28,1)
labels_test = data_test[:, -1]

i = np.random.randint(0,images_train_val.shape[0])
plt.figure()
plt.imshow(images_train_val[i, :, :, 0])
plt.show()
print(labels_train_val[i])


os.makedirs(os.path.join(out_dir, mnist12k_dir), exist_ok=True)
np.savez(os.path.join(out_dir, mnist12k_dir, 'train.npz'), data=images_train_val[:10000], labels=labels_train_val[:10000])
np.savez(os.path.join(out_dir, mnist12k_dir, 'valid.npz'), data=images_train_val[10000:], labels=labels_train_val[10000:])
np.savez(os.path.join(out_dir, mnist12k_dir, 'test.npz'), data=images_test, labels=labels_test)


##################################################
##################################################
##################################################


data_train_val = np.loadtxt(os.path.join(mnist_rot_dir, 'mnist_all_rotation_normalized_float_train_valid.amat'))
data_test = np.loadtxt(os.path.join(mnist_rot_dir, 'mnist_all_rotation_normalized_float_test.amat'))


images_train_val = data_train_val[:, :-1].reshape(-1,28,28,1)
images_train_val = np.transpose(images_train_val, (0,2,1,3))
labels_train_val = data_train_val[:, -1]

images_test = data_test[:, :-1].reshape(-1,28,28,1)
images_test = np.transpose(images_test, (0,2,1,3))
labels_test = data_test[:, -1]

i = np.random.randint(0,images_train_val.shape[0])
plt.figure()
plt.imshow(images_train_val[i, :, :, 0])
plt.show()
print(labels_train_val[i])

os.makedirs(os.path.join(out_dir, mnist_rot_dir), exist_ok=True)
np.savez(os.path.join(out_dir, mnist_rot_dir, 'train.npz'), data=images_train_val[:10000], labels=labels_train_val[:10000])
np.savez(os.path.join(out_dir, mnist_rot_dir, 'valid.npz'), data=images_train_val[10000:], labels=labels_train_val[10000:])
np.savez(os.path.join(out_dir, mnist_rot_dir, 'test.npz'), data=images_test, labels=labels_test)
