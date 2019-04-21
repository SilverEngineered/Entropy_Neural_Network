import tensorflow as tf
from tqdm import tqdm
import utils
import numpy as np
import time

class fcn(object):
    def __init__(self,sess,args):
        self.sess = sess
        self.datapath = args.datapath
        self.epoch = args.epoch
        self.lr = args.learning_rate
        self.bs = args.batch_size
        self.img_shape = args.img_shape
        self.num_classes = args.num_classes
        self.build()

    def build(self):
        """ Build the network by assembling the architecture and assign placeholders """
        self.x_train, self.x_test, self.y_train, self.y_test = utils.getMNIST(self.datapath)
        #print(self.x_train[0].shape)
        #print(self.x_test[0].shape)
        #print(self.y_train[0].shape)
        #print(self.y_test[0].shape)
        self.x = tf.placeholder(tf.float32,[None,self.img_shape[0]*self.img_shape[1]*self.img_shape[2]])
        self.y = tf.placeholder('int32', [None])
        flattened = tf.contrib.layers.flatten(self.x)
        dense_layer1 = tf.layers.dense(flattened, 200)
        dense_layer2 = tf.layers.dense(dense_layer1, 10)
        self.y_ = tf.nn.softmax(dense_layer2)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=dense_layer2,labels=tf.one_hot(self.y,self.num_classes)))
        self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.loss)        
        self.correct = tf.equal(tf.argmax(tf.one_hot(tf.cast(self.y, 'int64'), self.num_classes), 1),
                   tf.argmax(tf.cast(self.y_, 'int64'), 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))
    def train(self,args):
        """ Train the network with the set arguments """
        self.sess.run(tf.global_variables_initializer())
        self.epoch_loss = []
        self.times = [0]
        self.accuracies = [self.test()]
        print("Training Starting")
        for epoch in tqdm(range(self.epoch)):
            epoch_loss = 0
            for j in range(int(len(self.x_train)/self.bs)-1):
                start_time = time.time()
                x_batch = self.x_train[j*self.bs:(j+1)*self.bs]    
                y_batch = self.y_train[j*self.bs:(j+1)*self.bs]    
                _, loss = self.sess.run([self.optim,self.loss],feed_dict={self.x: x_batch, self.y: y_batch})            
                epoch_loss += loss
                self.times.append(time.time()-start_time)
                self.accuracies.append(self.test())
            self.epoch_loss.append(epoch_loss)
    def test(self,per=.5):
        """ Test the network over testing set

            Inputs;

                     per (float) -> percentage of test dataset to use
            Outputs:
                     accuracy (float) -> current accuracy value
        """
        accuracies = []
        x_test = self.x_test[int(per*len(self.x_test)):]
        y_test = self.y_test[int(per*len(self.y_test)):]
        for j in range(int(len(x_test)/self.bs)-1):
            x_batch = x_test[j*self.bs:(j+1)*self.bs]    
            y_batch = y_test[j*self.bs:(j+1)*self.bs]    
            a = self.sess.run(self.accuracy, feed_dict={self.x: x_batch, self.y: y_batch}) 
            accuracies.append(a)
        print(np.mean(accuracies))
        return np.mean(accuracies)      
    def getTimeAndAcc():
        """ Return the times and accuracies of training """
        return self.times, self.accuracies
