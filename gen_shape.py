"""
Trains the IH-GAN and generates unit cell shapes based on the optimal material properties

Author(s): Wei Chen (wchen459@gmail.com)
"""

import argparse
import numpy as np
import scipy.io as sio

from ihgan import Model
from utils import ElapsedTimer, create_dir


if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser(description='Shape generation')
    parser.add_argument('mode', type=str, default='train', help='train or evaluate')
    args = parser.parse_args()
    assert args.mode in ['train', 'evaluate']
    
    # Hyperparameters for GAN
    noise_dim = 3
    train_steps = 5000
    save_interval = 0
    batch_size = 32
    
    # Read data
    X = np.load('data/dvar.npy') # design variables
    C = np.load('data/mat_prp.npy') # material properties
    
    # Train/test split
    N = X.shape[0]
    split = int(0.8 * N)
    X_train = X[:split]
    C_train = C[:split]
    X_test = X[split:]
    C_test = C[split:]
    
    # Create a folder for the trained model
    model_dir = 'trained_model'
    create_dir(model_dir)
    
    # Train GAN
    model = Model(noise_dim, X.shape[1], C.shape[1], n_classes=3)
    if args.mode == 'train':
        timer = ElapsedTimer()
        model.train(X_train, C_train, X_test, C_test, batch_size=batch_size, train_steps=train_steps, 
                    save_interval=save_interval, save_dir=model_dir)
        elapsed_time = timer.elapsed_time()
        runtime_mesg = 'Wall clock time for training: %s' % elapsed_time
        print(runtime_mesg)
    else:
        model.restore(save_dir=model_dir)
        
    # Generate synthetic design variables given target material properties
    opt_dir = 'opt'
    C_tgt = sio.loadmat('{}/tgt_prp.mat'.format(opt_dir))['prop']
    X_synth = model.synthesize(C_tgt)
    np.save('{}/dvar_synth.npy'.format(opt_dir), X_synth)
    sio.savemat('{}/dvar_synth.mat'.format(opt_dir), {'vect':X_synth})
    