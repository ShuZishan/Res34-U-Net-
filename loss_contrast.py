import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"\\model\\default\\"
loss_Adam_learning_rate = []
for i in range(1, 4):
    f = open(path+"loss_Adam_learning_rate_0.00{0}.pkl".format(i), 'rb')
    loss_Adam_learning_rate.append(np.array(pickle.load(f)))
loss_RMSProp_learning_rate = []
for i in range(1, 4):
    f = open(path+"loss_RMSProp_learning_rate_0.00{0}.pkl".format(i), 'rb')
    loss_RMSProp_learning_rate.append(np.array(pickle.load(f)))
plt.title("loss contrast")
for index, i in enumerate(loss_Adam_learning_rate):
    plt.plot(i, label="Adam(learning_rate=0.00{0})".format(index+1))
for index, i in enumerate(loss_RMSProp_learning_rate):
    plt.plot(i, label="RMSProp(learning_rate=0.00{0})".format(index+1))
legend_font = {
    'weight': 'normal',
    'size': 7,
}
plt.legend(loc='upper right', prop=legend_font)
plt.xlabel("x - data size")
plt.ylabel("y - loss value")
plt.savefig('loss_contrast.png')
plt.show()
