import tensorflow as tf
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--datapath', default='../data')
parser.add_argument('--epoch', type=int,default=100)
parser.add_argument('--learning_rate',default=10e-4)
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
