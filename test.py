import os
import struct
import numpy as np
from scipy.special import expit
import random
import math
import matplotlib.pyplot as plt

BATCH_SIZE = 64
W1 = np.random.randn(784,196)
b1 = np.random.randn(196,1)
W2 = np.random.randn(196,10)
b2 = np.random.randn(10,1)

accu = np.random.randn(5000)

def load():
	global W1
	global b1
	global W2
	global b2
	global accu
	W1 = np.load("./model2/W1.npy")
	b1 = np.load("./model2/b1.npy")
	W2 = np.load("./model2/W2.npy")
	b2 = np.load("./model2/b2.npy")
	accu = np.load("./model2/accu.npy")
def process(x):
  global W1
  global b1
  out1 = expit(np.dot(W1.T,x)+b1)
  y = expit(np.dot(W2.T,out1)+b2)
  return y

def load_mnist():
  #"""Load MNIST data from `path`"""
  path = "./data/"
  kind = "test"
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

def maxindex(y):
  index = 0
  maxx = 0
  for i in range(10):
    if y[i][0]>maxx:
      maxx = y[i][0]
      index = i
  return index

def perdict(testX,testY):
	acc = 0
	for i in range(testX.shape[0]):
		x = testX[i].reshape(784,1)
		y = process(x)
		y_ = testY[i].reshape(10,1)
		if maxindex(y)==maxindex(y_):
			acc = acc+1
	acc = acc/10000
	print("Accuracy on testset is %f" % acc)

def draw():
	global accu
	x = []
	for i in range(5000):
		x.append(20*i)
	y = []
	for i in range(5000):
		y.append(float(accu[i]))
	plt.plot(x,y)
	plt.show()
	#print("huaa")
def draw_a_pic(x):
	y = maxindex(process(x))
	print("predict result: %d"%y)
	plt.imshow(x.reshape(28,28), cmap='Greys', interpolation='nearest')
	plt.show()
	
	


def main():
	load()
	testX,testY = load_mnist()
	print("Size of test set is"+str(testX.shape[0]))
	perdict(testX,testY)
	temp = 0
	while 1:
		a = input('please in put a number between 0-9999 or -1 to quit: ')
		temp = int(a)
		if temp==-1:
			break
		draw_a_pic(testX[temp].reshape(784,1))
	
	#
	#draw()
if __name__ == "__main__":
  main()


