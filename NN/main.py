import tensorflow as tf
import argparse
from model import fcn
import os


parser = argparse.ArgumentParser()
parser.add_argument('--datapath', default=os.path.join("..","data","MNIST","full_data",""))
parser.add_argument('--epoch', type=int,default=200)
parser.add_argument('--learning_rate',type=float,default=.00001)
parser.add_argument('--img_dims', type=int,default=784)
parser.add_argument('--num_classes', type=int,default=10)
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args()

def main(_):
    if not os.path.exists(args.datapath):
        raise Exception('Input datapath does not exist')
    with tf.Session() as sess:
        model = fcn(sess,args)
        model.train(args)
        model.saveOutput()
if __name__ == '__main__':
    tf.app.run()
