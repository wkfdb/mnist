import os
import struct
import numpy as np
from scipy.special import expit
import random

BATCH_SIZE = 64
W1 = np.random.randn(784,196)
b1 = np.random.randn(196,1)
W2 = np.random.randn(196,10)
b2 = np.random.randn(10,1)

accu = np.random.randn(5000)

A = 0.1#learning rate
dec_rate = 1000

def saver():
  global W1
  global b1
  global W2
  global b2
  global accu
  np.save("./model2/W1.npy",W1)
  np.save("./model2/b1.npy",b1)
  np.save("./model2/W2.npy",W2)
  np.save("./model2/b2.npy",b2)
  np.save("./model2/accu.npy",accu)
  #l+b = np.load("filename.npy")


def load_mnist(path, kind='train'):
  #"""Load MNIST data from `path`"""
  labels_path = os.path.join(path,'%s-labels.idx1-ubyte' % kind)
  images_path = os.path.join(path,'%s-images.idx3-ubyte' % kind)
  with open(labels_path, 'rb') as lbpath:
    magic, n = struct.unpack('>II',lbpath.read(8))
    labels = np.fromfile(lbpath,dtype=np.uint8)

    label = np.zeros((len(labels),10),dtype=np.float32)
    for i in range(len(labels)):
    	label[i][labels[i]] = 1


  with open(images_path, 'rb') as imgpath:
    magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
    images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
    image = np.array(images,dtype=np.float32)

  return image, label


def create_dataset():
	trainX,trainY = load_mnist('./data/','train')
	testX,testY=load_mnist('./data/','test')
	return trainX,trainY,testX,testY
  #images为60000*784维数组，代表了60000张图片，每张图片的每个像素为一个灰度值，用一维向量保存
  #labels为60000个标签，





def process(x):
  global W1
  global W2
  global b1
  global b2
  out11 = np.dot(W1.T,x)+b1
  out1 = expit(out11)
  y = expit(np.dot(W2.T,out1)+b2)
  return y,out1,out11

#反向传播过程
def back_process(y,y_,x,out1,out11):
  global W1
  global W2
  global b1
  global b2
  global A
  dz2 = y-y_
  dw2 = np.dot(out1,dz2.T)
  db2 = dz2
  dz1 = np.dot(W2,dz2)
  dw1 = np.dot(x,dz1.T)
  db1 = dz1
  W1 = W1-np.dot(A,dw1)
  b1 = b1 - np.dot(A,db1)
  W2 = W2-np.dot(A,dw2)
  b2 = b2 - np.dot(A,db2)

def loss(y,y_):
  summ = 0
  for i in range(10):
    summ = summ + (y_[i][0]-y[i][0])*(y_[i][0]-y[i][0])
  return summ/2

def maxindex(y):
  index = 0
  maxx = 0
  for i in range(10):
    if y[i][0]>maxx:
      maxx = y[i][0]
      index = i
  return index

def train():
  global accu
  global A
  global dec_rate
  trainX,trainY,testX,testY=create_dataset()
  for i in range(100000):
    j = random.randint(0,59999)
    x = trainX[j].reshape(784,1)
    y,out1,out11=process(x)
    back_process(y,trainY[j].reshape(10,1),x,out1,out11)

    if i % 20 == 0:
      acc = 0
      losss = 0
      for j in range(1000):
        k = random.randint(0,9999)
        y,_,_=process(testX[k].reshape(784,1))
        y_=testY[k].reshape(10,1)
        losss = losss+loss(y,y_)/1000

        if maxindex(y) == maxindex(y_):
          acc = acc+0.001
      print("%d samples trained, loss is %f, accuracy on test is %f, learning rate is %f" % (i,losss,acc,A))
      index = int(i/20)
      accu[index]=acc
    if i % dec_rate == 0 and i!=0:
      A = A*0.9
      if A < 0.0001:
        A = 0.0001


def main():
  
  
  train()
  saver()
  #np.save("filename.npy",a)
  #l+b = np.load("filename.npy")

if __name__ == "__main__":
  main()
