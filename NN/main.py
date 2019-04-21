import tensorflow as tf
import argparse
from model import fcn
import os


parser = argparse.ArgumentParser()
parser.add_argument('--datapath', default='../data')
parser.add_argument('--epoch', type=int,default=5)
parser.add_argument('--learning_rate',default=10e-4)
parser.add_argument('--img_shape', default=[28,28,1])
parser.add_argument('--num_classes', type=int,default=10)
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()

def main(_):
    if not os.path.exists(args.datapath):
        raise Exception('Input datapath does not exist')
    with tf.Session() as sess:
        model = fcn(sess,args)
        model.train(args)
if __name__ == '__main__':
    tf.app.run()
