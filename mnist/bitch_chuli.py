import sys, os
sys.path.append(os.pardir)
from neuralnet_mnist import get_data, init_network, predict, accuracy_cnt
import numpy as np
x,_=get_data()
network=init_network()
W1,W2,W3=network['W1'],network['W2'],network['W3']
print(x.shape)
print(x[0].shape)
print(W1.shape,W2.shape,W3.shape)

x,t=get_data()
network=init_network()
batch_size=100
accurary_cnt=0

for i in range(0,len(x),batch_size):
    x_batch=x[i:i+batch_size]
    y_batch=predict(network,x_batch)
    p=np.argmax(y_batch,axis=1)
    accurary_cnt+=np.sum(p==t[i:i+batch_size])
print("accuracy:"+str(float(accuracy_cnt)/len(x)))


