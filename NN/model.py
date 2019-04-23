import tensorflow as tf
from tqdm import tqdm
import utils
import numpy as np
import time
import csv

class fcn(object):
    def __init__(self,sess,args):
        self.sess = sess
        self.datapath = args.datapath
        self.epoch = args.epoch
        self.lr = args.learning_rate
        self.bs = args.batch_size
        self.img_dims = args.img_dims
        self.num_classes = args.num_classes
        self.trial = args.trial
        self.build()
    def build(self):
        """ Build the network by assembling the architecture and assign placeholders """
        self.x_train, self.x_test, self.y_train, self.y_test = utils.getMNIST(self.datapath)
        self.x = tf.placeholder(float,[None,self.img_dims])
        self.y = tf.placeholder('int32', [None])
        layer1 = tf.layers.dense(self.x,256,activation=tf.nn.relu)
        layer2 = tf.layers.dense(layer1,256,activation=tf.nn.relu)
        final_layer = tf.layers.dense(layer2, self.num_classes)
        self.y_ = tf.nn.softmax(final_layer)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.y_,labels=tf.one_hot(self.y,self.num_classes)))
        self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.choice=tf.argmax(tf.cast(self.y_,'int32'),1)
        self.real = tf.argmax(tf.one_hot(self.y,self.num_classes),1)        
        self.correct = tf.equal(self.choice,self.real)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))
    def train(self,args):
        """ Train the network with the set arguments """
        self.sess.run(tf.global_variables_initializer())
        self.epoch_loss = self.sess.run([self.loss],feed_dict={self.x: self.x_train[:self.bs],self.y: self.y_train[:self.bs]})          
        self.accuracies = [self.test()]
        print("Training Starting")
        for epoch in tqdm(range(self.epoch)):
            epoch_loss = []
            for j in range(int(len(self.x_train)/self.bs)-1):
                x_batch = self.x_train[j*self.bs:(j+1)*self.bs]    
                y_batch = self.y_train[j*self.bs:(j+1)*self.bs]    
                _, loss = self.sess.run([self.optim,self.loss],feed_dict={self.x: x_batch, self.y: y_batch})            
                epoch_loss.append(loss)
            self.epoch_loss.append(np.mean(epoch_loss))
            self.accuracies.append(self.test())
    def test(self):
        """ Test the network over testing set
            Outputs:
                     accuracy (float) -> current accuracy value
        """
        accuracies = []
        x_test = self.x_test
        y_test = self.y_test
        for j in range(int(len(x_test)/self.bs)-1):
            x_batch = x_test[j*self.bs:(j+1)*self.bs]    
            y_batch = y_test[j*self.bs:(j+1)*self.bs]    
            y_=self.sess.run(self.y_,{self.x: x_batch})
            choice=self.sess.run(self.choice,{self.x: x_batch})
            real=self.sess.run(self.real,{self.y: y_batch})
            a = self.sess.run(self.accuracy, feed_dict={self.x: x_batch, self.y: y_batch}) 
            accuracies.append(a)
        print(np.mean(accuracies))
        return np.mean(accuracies)      
    def saveOutput(self):
        """ Save accuracies, losses"""
        f=open(self.datapath + "nnacc" + self.trial + ".csv",'w')
        for i in self.accuracies:
            f.write(str(i))
            f.write("\n")
        f.close()
        f=open(self.datapath + "nnloss" +self.trial +  ".csv",'w')
        for i in self.epoch_loss:
            f.write(str(i))
            f.write("\n")
        f.close()
