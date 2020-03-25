from experiment_builder import ExperimentBuilder
from model_architectures import *
from torch.utils.data import DataLoader
import data_providers as data_providers
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_result_graphs(plot_name, enn_stats, pnn_stats):
    fig_1 = plt.figure(figsize=(8, 4))
    ax_1 = fig_1.add_subplot(111)    
    line1 = ax_1.plot(np.arange(0, len(enn_stats['train_loss'])), 
                      enn_stats['train_loss'], label='enn_train_loss')
    line2 = ax_1.plot(np.arange(0, len(enn_stats['val_loss'])), 
                      enn_stats['val_loss'], label='enn_val_loss')

    ax_1r = ax_1.twinx()
    line3 = ax_1r.plot(np.arange(0, len(pnn_stats['train_loss'])), 
                       pnn_stats['train_loss'], '-m', label='pnn_train_loss')
    line4 = ax_1r.plot(np.arange(0, len(pnn_stats['val_loss'])), 
                        pnn_stats['val_loss'], '-k', label='pnn_val_loss')
    lns = line1 + line2 + line3 + line4
    labs = [l.get_label() for l in lns]
    
    ax_1.legend(lns, labs, loc=0)        
    ax_1.set_ylabel('Loss')
    ax_1.set_xlabel('Epoch number')
    ax_1.grid(False)
    ax_1r.grid(False)

    # Plot the change in the validation and training set accuracy over training.
    fig_2 = plt.figure(figsize=(8, 4))
    ax_2 = fig_2.add_subplot(111)
    
    ax_2.plot(np.arange(0, len(enn_stats['train_acc'])), 
              enn_stats['train_acc'], label='enn_train_acc')
    ax_2.plot(np.arange(0, len(enn_stats['val_acc'])), 
              enn_stats['val_acc'], label='enn_val_acc')
    ax_2.plot(np.arange(0, len(pnn_stats['train_acc'])), 
              pnn_stats['train_acc'], '-m', label='pnn_train_acc')
    ax_2.plot(np.arange(0, len(pnn_stats['val_loss'])), 
              pnn_stats['val_acc'], '-k', label='pnn_val_acc')
                    
    ax_2.legend(loc=0)
    ax_2.set_ylabel('Accuracy')
    ax_2.set_xlabel('Epoch number')
    
    fig_1.savefig('./data/{}_loss_performance.pdf'.format(plot_name), dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='pdf',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
    
    fig_2.savefig('./data/{}_accuracy_performance.pdf'.format(plot_name), dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='pdf',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
    
    
def plot_pnn_graphs(plot_name, pnn_stats):
    fig_1 = plt.figure(figsize=(8, 4))
    ax_1 = fig_1.add_subplot(111)    
    
    ax_1.plot(np.arange(0, len(pnn_stats['train_loss'])), 
                       pnn_stats['train_loss'], label='pnn_train_loss')
    ax_1.plot(np.arange(0, len(pnn_stats['val_loss'])), 
                        pnn_stats['val_loss'], label='pnn_val_loss')
    
    ax_1.legend(loc=0)        
    ax_1.set_ylabel('Loss')
    ax_1.set_xlabel('Epoch number')
    ax_1.grid(False)


    # Plot the change in the validation and training set accuracy over training.
    fig_2 = plt.figure(figsize=(8, 4))
    ax_2 = fig_2.add_subplot(111)
    
    ax_2.plot(np.arange(0, len(pnn_stats['train_acc'])), 
              pnn_stats['train_acc'], label='pnn_train_acc')
    ax_2.plot(np.arange(0, len(pnn_stats['val_loss'])), 
              pnn_stats['val_acc'], label='pnn_val_acc')
                    
    ax_2.legend(loc=0)
    ax_2.set_ylabel('Accuracy')
    ax_2.set_xlabel('Epoch number')
    
    fig_1.savefig('./data/{}_loss_performance.pdf'.format(plot_name), dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='pdf',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
    
    fig_2.savefig('./data/{}_accuracy_performance.pdf'.format(plot_name), dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='pdf',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
    
    
def plot_acc_recall_graphs(plot_name, accs, recalls):    
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.plot(np.arange(0, accs.shape[0]), accs, label='accuracy')
    ax.plot(np.arange(0, recalls.shape[0]), recalls, '--', label='recall')
    
    ax.legend(loc=0)
    ax.set_ylabel('accuracy & recall')
    ax.set_xlabel(plot_name)
    
    fig.savefig('./data/{}_acc_recall.pdf'.format(plot_name), dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='pdf',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

if __name__ == '__main__':  
    batch_size = 100
    num_epochs = 3
    weight_decay_coefficient = 0
    lr = 0.001
    use_gpu = True
    
    use_enn = True
    
    enn_input_dim = 52 + 2 + 318
    enn_output_dim = 52
    enn_hidden_dim = 1500
    enn_layers = 8
    
    if use_enn:
        pnn_input_dim = enn_input_dim + enn_output_dim
    else:
        pnn_input_dim = enn_input_dim
    pnn_output_dim = 38
    pnn_hidden_dim = 1200
    pnn_layers = 10
    
    train_data = data_providers.BridgeData(root='data', set_name='train')
    val_data = data_providers.BridgeData(root='data', set_name='val')
    test_data = data_providers.BridgeData(root='data', set_name='test')
    
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)
    
    if use_enn:
        enn = ENN(
                input_shape=(batch_size, enn_input_dim),
                num_output_classes=enn_output_dim,
                num_filters=enn_hidden_dim,
                num_layers=enn_layers)
    else:
        enn = None
    
    pnn = PNN(
            input_shape=(batch_size, pnn_input_dim),
            num_output_classes=pnn_output_dim,
            num_filters=pnn_hidden_dim,
            num_layers=pnn_layers)
    
    bridge_experiment = ExperimentBuilder(estimation_model=enn,
                                        policy_model=pnn,
                                        num_epochs=num_epochs,
                                        weight_decay_coefficient=weight_decay_coefficient,
                                        use_gpu=use_gpu,
                                        train_data=train_data_loader, val_data=val_data_loader,
                                        test_data=test_data_loader,
                                        lr=lr)  # build an experiment object
    
    if use_enn:
        enn_stats, pnn_stats = bridge_experiment.run_experiment()
        plot_result_graphs('baseline', enn_stats, pnn_stats)
    else:
        pnn_stats = bridge_experiment.run_experiment()
        plot_pnn_graphs('only_pnn', pnn_stats)
    
    if use_enn:
        enn_accs, enn_recalls = bridge_experiment.eval_ordered_card()
        plot_acc_recall_graphs('ordered_card', enn_accs, enn_recalls)
    
    pnn_accs, pnn_recalls = bridge_experiment.eval_ordered_bid()
    plot_acc_recall_graphs('ordered_bid', pnn_accs, pnn_recalls)