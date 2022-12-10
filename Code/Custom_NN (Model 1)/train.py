# Author: Ruinan Ma
# Email: r7ma@ucsd.edu
# This file contains essential functions for training

import copy
from neuralnet import *
from util import *

from tqdm import tqdm

def train(model, x_train, y_train, x_valid, y_valid, config, config_file_name):
    # Read in the esssential configs
    layer_specs = config['layer_specs']
    activation = config['activation']
    learning_rate = config['learning_rate']
    batchsize = config['batch_size']
    epochs = config['epochs']
    early_stop = config['early_stop']
    early_stop_epoch = config['early_stop_epoch']
    L2_panalty = config['L2_penalty']
    momentum = config['momentum']
    momentum_gamma = config['momentum_gamma']
    weight_type = config['weight_type']

    num_val_no_improve = 0
    val_acc_prev_epoch = 0
    val_acc_best = 0
    
    model_best = None

    total_train_loss = []
    total_train_acc = []
    total_val_loss = []
    total_val_acc = []

    for epoch in tqdm(range(1, epochs + 1)):
        epoch_train_loss = 0
        epoch_train_acc = 0

        for minibatch in generate_minibatches((x_train, y_train), batchsize):
                
                x_batch, y_batch = minibatch
                
                # Forward pass
                loss, acc = model.forward(x_batch.T, y_batch.T)
                
                # Backward pass
                model.backward()
                
                epoch_train_loss += loss
                epoch_train_acc += acc
        
        epoch_train_loss /= len(x_train)
        epoch_train_acc /= len(x_train)
        tqdm.write('Epoch {}: Train loss: {}, Train accuracy: {}, learning_rate: {} ,momentum_gamma: {} '.format(epoch, epoch_train_loss, epoch_train_acc, model.learning_rate, model.momentum_gamma))
        epoch_val_loss, epoch_val_acc = model(x_valid.T, y_valid.T)
        epoch_val_loss /= x_valid.shape[0]
        epoch_val_acc /= y_valid.shape[0]
        tqdm.write('          Val loss: {}, Val accuracy: {}, learning_rate: {},  momentum_gamma: {} '.format(epoch_val_loss, epoch_val_acc, model.learning_rate, model.momentum_gamma))
        model.learning_rate /= 1.12
        model.momentum_gamma *= 1.04
        total_train_loss.append(epoch_train_loss)
        total_train_acc.append(epoch_train_acc)
        total_val_loss.append(epoch_val_loss)
        total_val_acc.append(epoch_val_acc)

        if epoch_val_acc < val_acc_prev_epoch:
            num_val_no_improve += 1
            if num_val_no_improve == early_stop_epoch:
                tqdm.write("Early stopping")
                break
        else:
            num_val_no_improve = 0
            val_acc_prev_epoch = epoch_val_acc
        
        # save the model with the best validation accuracy
        if epoch_val_acc > val_acc_best:
            val_acc_best = epoch_val_acc
            model_best = copy.deepcopy(model)

    plot_loss_acc(total_train_loss, total_train_acc, total_val_loss, total_val_acc, config_file_name)

    return model_best


def modelTest(model, X_test, y_test):
    loss, acc = model(X_test.T, y_test.T)
    return acc, loss


def plot_loss_acc(total_train_loss, total_train_acc, total_val_loss, total_val_acc, config_file_name):
    config_file_name = config_file_name.split('.')[0]

    fig = plt.figure(figsize=(10, 5))
    plt.plot(total_train_loss, label='train loss')
    plt.plot(total_val_loss, label='val loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.savefig('output/' + config_file_name + '_loss.png')
    plt.close(fig)

    fig = plt.figure(figsize=(10, 5))
    plt.plot(total_train_acc, label='train acc')
    plt.plot(total_val_acc, label='val acc')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.savefig('output/' + config_file_name + '_acc.png')
    plt.close(fig)

    