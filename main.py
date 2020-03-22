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
    for name in ['enn', 'pnn']:
        for k in ['train_loss', 'val_loss']:
            if name is 'enn':
                item = enn_stats[k]
                ax_1.plot(np.arange(0, len(item)), 
                          item, label='{}_{}'.format(name, k))
            else:
                item = pnn_stats[k]
                ax_1r = ax_1.twinx()
                ax_1r.plot(np.arange(0, len(item)), 
                           item, 'r', label='{}_{}'.format(name, k))
            
    ax_1.legend(loc=0)
    ax_1.set_ylabel('Loss')
    ax_1.set_xlabel('Epoch number')

    # Plot the change in the validation and training set accuracy over training.
    fig_2 = plt.figure(figsize=(8, 4))
    ax_2 = fig_2.add_subplot(111)
    for name in ['enn', 'pnn']:
        for k in ['train_acc', 'val_acc']:
            if name is 'enn':
                item = enn_stats[k]
            else:
                item = pnn_stats[k]
            ax_2.plot(np.arange(0, len(item)), 
                      item, label='{}_{}'.format(name, k))
            
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

if __name__ == '__main__':
    batch_size = 100
    num_epochs = 5
    weight_decay_coefficient = 0
    use_gpu = True
    
    enn_input_dim = 52 + 2 + 318
    enn_output_dim = 52
    enn_hidden_dim = 1500
    enn_layers = 8
    
    pnn_input_dim = enn_input_dim + enn_output_dim
    #pnn_input_dim = enn_input_dim
    pnn_output_dim = 38
    pnn_hidden_dim = 1200
    pnn_layers = 10
    
    train_data = data_providers.BridgeData(root='data', set_name='train')
    val_data = data_providers.BridgeData(root='data', set_name='val')
    test_data = data_providers.BridgeData(root='data', set_name='test')
    
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)
    
    enn = ENN(
            input_shape=(batch_size, enn_input_dim),
            num_output_classes=enn_output_dim,
            num_filters=enn_hidden_dim,
            num_layers=enn_layers)
    
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
                                        test_data=test_data_loader)  # build an experiment object
    
    enn_states, pnn_stats = bridge_experiment.run_experiment()
    
    plot_result_graphs('baseline', enn_states, pnn_stats)