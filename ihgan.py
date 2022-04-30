"""
IH-GAN

Author(s): Wei Chen (wchen459@gmail.com)
"""

import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from data_processing import Normalizer
from data_processing import postprocess_design_variables, preprocess_design_variables, preprocess_material_properties


EPSILON = 1e-7

class Model(object):
    
    def __init__(self, dim_Z, dim_X, dim_C, n_classes=3):

        self.dim_Z = dim_Z
        self.dim_X = dim_X
        self.dim_C = dim_C
        self.n_classes = n_classes
        
    def generator(self, z, c, reuse=tf.AUTO_REUSE):
        
        with tf.variable_scope('Generator', reuse=reuse):
            
            zc = tf.concat([z, c], axis=1)
            
            x = tf.layers.dense(zc, 128)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            
            x = tf.layers.dense(x, 256)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            
            x = tf.layers.dense(x, 512)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            
            x = tf.layers.dense(x, 1024)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            
            # alpha1, alpha2, alpha3
            x_alpha = tf.layers.dense(x, self.n_classes)
            x_alpha = tf.nn.softmax(x_alpha)
            
            # t1, t2, t3
            x_t = tf.layers.dense(x, self.dim_X-self.n_classes)
                
            x = tf.concat([x_alpha, x_t], axis=1, name='gen')
            
            return x
        
    def discriminator(self, x, c, reuse=tf.AUTO_REUSE):
        
        with tf.variable_scope('Discriminator', reuse=reuse):
            
            xc = tf.concat([x, c], axis=1)
            
            y = tf.layers.dense(xc, 1024)
            y = tf.nn.leaky_relu(y, alpha=0.2)
            
            y = tf.layers.dense(y, 512)
            y = tf.nn.leaky_relu(y, alpha=0.2)
            
            y = tf.layers.dense(y, 256)
            y = tf.nn.leaky_relu(y, alpha=0.2)
            
            y = tf.layers.dense(y, 128)
            y = tf.nn.leaky_relu(y, alpha=0.2)
            
            y = tf.layers.dense(y, 1)
            
            return y
        
    def predictor(self, x, reuse=tf.AUTO_REUSE):
        
        with tf.variable_scope('Predictor', reuse=reuse):
            
            y = tf.layers.dense(x, 1024)
            y = tf.nn.leaky_relu(y, alpha=0.2)
            
            y = tf.layers.dense(y, 512)
            y = tf.nn.leaky_relu(y, alpha=0.2)
            
            y = tf.layers.dense(y, 256)
            y = tf.nn.leaky_relu(y, alpha=0.2)
            
            y = tf.layers.dense(y, 128)
            y = tf.nn.leaky_relu(y, alpha=0.2)
            
            y = tf.layers.dense(y, self.dim_C)
            
            return y
        
    def train(self, X_train, C_train, X_test, C_test, train_steps=10000, batch_size=32, 
              save_interval=0, save_dir='.'):
        
        print('Training model ...')
    
        # Normalize training data
        self.normalizer_X = Normalizer(data=X_train[:,self.n_classes:])
        self.normalizer_C = Normalizer(data=C_train)
        np.save('{}/bounds_dvar.npy'.format(save_dir), self.normalizer_X.bounds)
        np.save('{}/bounds_mat_prp.npy'.format(save_dir), self.normalizer_C.bounds)
        
        X_train_ = preprocess_design_variables(X_train, self.normalizer_X)
        X_test_ = preprocess_design_variables(X_test, self.normalizer_X)
        C_train_ = preprocess_material_properties(C_train, self.normalizer_C)
        C_test_ = preprocess_material_properties(C_test, self.normalizer_C)
        
        # Inputs
        self.x = tf.placeholder(tf.float32, shape=[None, self.dim_X], name='data')
        self.z = tf.placeholder(tf.float32, shape=[None, self.dim_Z], name='noise')
        self.c = tf.placeholder(tf.float32, shape=[None, self.dim_C], name='condition')
        
        # Outputs
        d_real = self.discriminator(self.x, self.c)
        self.x_fake = self.generator(self.z, self.c)
        d_fake = self.discriminator(self.x_fake, self.c)
        c_pred_real = self.predictor(self.x)
        c_pred_fake = self.predictor(self.x_fake)
        
        # Losses
        # Cross entropy losses for D
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))
        # Cross entropy losses for G
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)))
        # L1 loss for c
        c_loss_real = tf.reduce_mean(tf.abs(c_pred_real-self.c))
        c_loss_fake = tf.reduce_mean(tf.abs(c_pred_fake-self.c))
        
        # Optimizers
        d_optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
        g_optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
        
        # Generator variables
        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
        # Discriminator variables
        dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
        # Predictor variables
        pred_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Predictor')
        
        # Training operations
        gamma = 20
        d_train = d_optimizer.minimize(d_loss_real+d_loss_fake+gamma*c_loss_real, var_list=[dis_vars, pred_vars])
        g_train = g_optimizer.minimize(g_loss+gamma*c_loss_fake, var_list=[gen_vars])
        
        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        
        # Create summaries to monitor losses
        tf.summary.scalar('D_loss_for_real', d_loss_real)
        tf.summary.scalar('D_loss_for_fake', d_loss_fake)
        tf.summary.scalar('G_loss', g_loss)
        tf.summary.scalar('C_loss_for_real', c_loss_real)
        tf.summary.scalar('C_loss_for_fake', c_loss_fake)
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
        # Start training
        self.sess = tf.Session()
        
        # Run the initializer
        self.sess.run(init)
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter('{}/logs'.format(save_dir), graph=self.sess.graph)
    
        for t in range(train_steps):
    
            ind = np.random.choice(X_train_.shape[0], size=batch_size, replace=False)
            x_batch = X_train_[ind]
            c_batch = C_train_[ind]
            noise = np.random.normal(scale=0.5, size=(batch_size, self.dim_Z))
            summary_str, _, _, dlr, dlf, gl, clr, clf = self.sess.run([merged_summary_op, d_train, g_train, 
                                                                       d_loss_real, d_loss_fake, g_loss, c_loss_real, c_loss_fake], 
                                                                      feed_dict={self.x: x_batch, self.z: noise, 
                                                                                 self.c: c_batch})
            
            summary_writer.add_summary(summary_str, t+1)
            
            # Show messages
            log_mesg = "%d: [D] real %f fake %f L1 %f" % (t+1, dlr, dlf, clr)
            log_mesg = "%s  [G] fake %f L1 %f" % (log_mesg, gl, clf)
            print(log_mesg)
            
            if save_interval>0 and (t+1)%save_interval==0 or t+1 == train_steps:
                # Save the variables to disk.
                save_path = saver.save(self.sess, '{}/model'.format(save_dir))
                print('Model saved in path: %s' % save_path)
                    
    def restore(self, save_dir='.'):
        
        print('Loading model ...')
        
        self.normalizer_X = Normalizer(bounds=np.load('{}/bounds_dvar.npy'.format(save_dir)))
        self.normalizer_C = Normalizer(bounds=np.load('{}/bounds_mat_prp.npy'.format(save_dir)))
        
        self.sess = tf.Session()
        # Load meta graph and restore weights
        saver = tf.train.import_meta_graph('{}/model.meta'.format(save_dir))
        saver.restore(self.sess, tf.train.latest_checkpoint('{}/'.format(save_dir)))
        
        # Access and create placeholders variables            
        graph = tf.get_default_graph()
        self.x = graph.get_tensor_by_name('data:0')
        self.z = graph.get_tensor_by_name('noise:0')
        self.c = graph.get_tensor_by_name('condition:0')
        self.x_fake = graph.get_tensor_by_name('Generator/gen:0')

    def synthesize(self, condition, noise=None):
        condition = preprocess_material_properties(condition, self.normalizer_C)
        if noise is None:
            noise = np.random.normal(scale=0.5, size=(condition.shape[0], self.dim_Z))
        X = self.sess.run(self.x_fake, feed_dict={self.z: noise, self.c: condition})
        X = postprocess_design_variables(X, self.normalizer_X)
        return X
    
    