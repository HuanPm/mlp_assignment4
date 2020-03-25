import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import tqdm
import time

class ExperimentBuilder(nn.Module):
    def __init__(self, estimation_model, policy_model, num_epochs, train_data, val_data,
                 test_data, weight_decay_coefficient, use_gpu, lr):
        """
        Initializes an ExperimentBuilder object. Such an object takes care of running training and evaluation of a deep net
        on a given dataset. It also takes care of saving per epoch models and automatically inferring the best val model
        to be used for evaluating the test set metrics.
        :param estimation_model: A pytorch nn.Module which implements a network architecture.
        :param policy_model: A pytorch nn.Module which implements a network architecture.
        :param num_epochs: Total number of epochs to run the experiment
        :param train_data: An object of the DataProvider type. Contains the training set.
        :param val_data: An object of the DataProvider type. Contains the val set.
        :param test_data: An object of the DataProvider type. Contains the test set.
        :param weight_decay_coefficient: A float indicating the weight decay to use with the adam optimizer.
        :param use_gpu: A boolean indicating whether to use a GPU or not.
        """
        super(ExperimentBuilder, self).__init__()
        
        self.enn = estimation_model
        self.pnn = policy_model
        
        if torch.cuda.device_count() > 1 and use_gpu:
            self.device = torch.cuda.current_device()
            if self.enn is not None:
                self.enn.to(self.device)
                self.enn = nn.DataParallel(module=self.enn)
            self.pnn.to(self.device)
            self.pnn = nn.DataParallel(module=self.enn)
            print('Use Multi GPU', self.device)
        elif torch.cuda.device_count() == 1 and use_gpu:
            self.device =  torch.cuda.current_device()
            if self.enn is not None:
                self.enn.to(self.device)  # sends the model from the cpu to the gpu
            self.pnn.to(self.device)  # sends the model from the cpu to the gpu
            print('Use GPU', self.device)
        else:
            print("use CPU")
            self.device = torch.device('cpu')  # sets the device to be CPU
            print(self.device)
        
        if self.enn is not None:
            self.enn.reset_parameters()  # re-initialize network parameters
        self.pnn.reset_parameters()  # re-initialize network parameters
        
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr, amsgrad=False,
                                    weight_decay=weight_decay_coefficient)
        self.learning_rate_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                            T_max=num_epochs,
                                                                            eta_min=0.00002)
             
        # Set best models to be at 0 since we are just starting
        if self.enn is not None:
            self.enn_best_val_model_idx = 0
            self.enn_best_val_model_acc = 0.
        self.pnn_best_val_model_idx = 0
        self.pnn_best_val_model_acc = 0.
        
        self.num_epochs = num_epochs
        self.criterion = nn.CrossEntropyLoss().to(self.device)  # send the loss computation to the GPU
        
        self.state = dict()
        
    def run_train_iter(self, x, y, z, model):
        """
        Receives the inputs and targets for the model and runs a training iteration. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape batch_size, num_features
        :param y: The targets for the enn model. A numpy array of shape batch_size, num_classes
        :param z: The targets for the pnn model. A numpy array of shape batch_size, num_classes
        :param model: The model currently use
        :return: the loss and accuracy for this batch
        """
        self.train()  # sets model to training mode (in case batch normalization or other methods have different procedures for training and evaluation)
        x, y, z = x.float().to(device=self.device), y.float().to(
            device=self.device), z.float().to(device=self.device) # send data to device as torch tensors
        
        if model is 'enn':
            out = self.enn.forward(x)  # forward the data in the enn
                      
            #loss = torch.mean(torch.sum(- y * F.log_softmax(out), 1))  # compute loss
            loss = 0
            for i in np.arange(out.shape[1]):
                loss += F.binary_cross_entropy(input=out[:, i], target=y[:, i]) # compute loss
        
            self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
            loss.backward(retain_graph=True)  # backpropagate to compute gradients for current iter loss
    
            self.learning_rate_scheduler.step(epoch=self.current_epoch)
            self.optimizer.step()  # update network parameters
                   
            # compute one-hot of max 13 probs of deals 
            _, predicted = torch.topk(out.data, 13, dim=1)
            onehot_predicted = torch.zeros(out.data.shape[0], out.data.shape[1], device=self.device)
            for i in np.arange(predicted.shape[0]):
                onehot_predicted[i, predicted[i]] = 1
                
            accuracy = np.mean(list(((onehot_predicted * y.data).sum(1) / 13).cpu()))  # compute accuracy
        
        elif model is 'pnn':
            z = torch.argmax(z, 1)
            
            if self.enn is not None:
                out = self.enn.forward(x)
                out = self.pnn.forward(torch.cat((x, out), 1))  # forward the data in the pnn
            else:
                out = self.pnn.forward(x)

            loss = F.cross_entropy(input=out, target=z) # compute loss
                
            self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
            loss.backward()  # backpropagate to compute gradients for current iter loss
        
            self.learning_rate_scheduler.step(epoch=self.current_epoch)
            self.optimizer.step()  # update network parameters
            
            _, predicted = torch.max(out.data, 1)  # get argmax of predictions
            
            accuracy = np.mean(list(predicted.eq(z.data).cpu()))  # compute accuracy
            
        return loss.cpu().data.numpy(), accuracy
    
    
    def run_evaluation_iter(self, x, y, z, model):
        """
        Receives the inputs and targets for the model and runs an evaluation iterations. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the enn model. A numpy array of shape batch_size, num_classes
        :param z: The targets for the pnn model. A numpy array of shape batch_size, num_classes
        :param model: The model currently use
        :return: the loss and accuracy for this batch
        """
        self.eval()  # sets the system to validation mode
        x, y, z = x.float().to(device=self.device), y.float().to(
            device=self.device), z.float().to(device=self.device)  # convert data to pytorch tensors and send to the computation device
        
        if model is 'enn':
            out = self.enn.forward(x)  # forward the data in the model
            loss = 0
            for i in np.arange(out.shape[1]):
                loss += F.binary_cross_entropy(input=out[:, i], target=y[:, i]) # compute loss
            
            # compute one-hot of max 13 probs of deals 
            _, predicted = torch.topk(out.data, 13, dim=1)
            onehot_predicted = torch.zeros(out.data.shape[0], out.data.shape[1], device=self.device)
            for i in np.arange(predicted.shape[0]):
                onehot_predicted[i, predicted[i]] = 1
            
            accuracy = np.mean(list(((onehot_predicted * y.data).sum(1) / 13).cpu()))  # compute accuracy    
            
        elif model is 'pnn':
            z = torch.argmax(z, 1)
            
            if self.enn is not None:
                out = self.enn.forward(x)
                out = self.pnn.forward(torch.cat((x, out), 1))  # forward the data in the pnn
            else:
                out = self.pnn.forward(x)
            
            loss = F.cross_entropy(input=out, target=z) # compute loss
                
            _, predicted = torch.max(out.data, 1)  # get argmax of predictions
            
            accuracy = np.mean(list(predicted.eq(z.data).cpu()))  # compute accuracy
            
        return loss.cpu().data.numpy(), accuracy
        
    
    def save_model(self, model_save_name, model_idx, best_validation_model_idx,
                   best_validation_model_acc):
        """
        Save the network parameter state and current best val epoch idx and best val accuracy.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :param best_validation_model_idx: The index of the best validation model to be stored for future use.
        :param best_validation_model_acc: The best validation accuracy to be stored for use at test time.
        :param state: The dictionary containing the system state.

        """
        self.state['network'] = self.state_dict()  # save network parameter and other variables.
        self.state['best_val_model_idx'] = best_validation_model_idx  # save current best val idx
        self.state['best_val_model_acc'] = best_validation_model_acc  # save current best val acc
        torch.save(self.state, f=os.path.join("model", "{}_{}".format(model_save_name, str(
            model_idx))))  # save state at prespecified filepath

    def load_model(self, model_save_name, model_idx):
        """
        Load the network parameter state and the best val model idx and best val acc to be compared with the future val accuracies, in order to choose the best val model
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :return: best val idx and best val model acc, also it loads the network state into the system state without returning it
        """
        state = torch.load(f=os.path.join("model", "{}_{}".format(model_save_name, str(model_idx))))
        self.load_state_dict(state_dict=state['network'])
        
        
    def eval_ordered_card(self):
        self.eval()  # sets the system to validation mode
        TP = np.zeros(52)
        FP = np.zeros(52)
        FN = np.zeros(52)
        TN = np.zeros(52)
        with tqdm.tqdm(total=len(self.test_data)) as pbar_test:
            for x, y, z in self.test_data:
                x, y = x.float().to(device=self.device), y.float().to(device=self.device)
                out = self.enn.forward(x)
                _, predicted = torch.topk(out.data, 13, dim=1)
            
                for i in np.arange(predicted.shape[0]):
                    for j in np.arange(52):
                        if j in predicted[i] and y[i][j] == 1:
                            TP[j] += 1
                        elif j in predicted[i] and y[i][j] == 0:
                            FP[j] += 1
                        elif j not in predicted[i] and y[i][j] == 1:
                            FN[j] += 1
                        else:
                            TN[j] += 1
                
                pbar_test.update(1)  # update progress bar status
                          
        acc = (TP + TN) / (TP + FP + FN + TN)
        recall = TP / (TP + FN)
        #precision = TP / (TP + FP)
        
        return acc, recall
    
    
    def eval_ordered_bid(self):
        self.eval()
        TP = np.zeros(38)
        FP = np.zeros(38)
        FN = np.zeros(38)
        TN = np.zeros(38)
        with tqdm.tqdm(total=len(self.test_data)) as pbar_test:
            for x, y, z in self.test_data:
                x, y, z = x.float().to(device=self.device), y.float().to(
                        device=self.device), z.float().to(device=self.device)  # convert data to pytorch tensors and send to the computation device
                
                z = torch.argmax(z, 1)
                if self.enn is not None:
                    out = self.enn.forward(x)
                    out = self.pnn.forward(torch.cat((x, out), 1))  # forward the data in the pnn
                else:
                    out = self.pnn.forward(x)
                _, predicted = torch.max(out.data, 1)  # get argmax of predictions
            
                for i in np.arange(predicted.shape[0]):
                    TN += np.ones(38)
                    
                    if predicted[i] == z[i]:
                        TN[z[i]] -= 1
                        TP[z[i]] += 1
                    else:
                        TN[predicted[i]] -= 1
                        TN[z[i]] -= 1
                        FP[predicted[i]] += 1
                        FN[z[i]] += 1
                    
                pbar_test.update(1)  # update progress bar status
                          
        acc = (TP + TN) / (TP + FP + FN + TN)
        recall = TP / (TP + FN)
        #precision = TP / (TP + FP)
        
        return acc, recall
        
        
    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        if self.enn is not None:
            # train enn model
            print("Starting to train enn")
            enn_total_losses = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []} 
            for i, epoch_idx in enumerate(range(self.num_epochs)):
                epoch_start_time = time.time()
                enn_current_epoch_losses = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}
                self.current_epoch = epoch_idx
                
                with tqdm.tqdm(total=len(self.train_data)) as pbar_train:  # create a progress bar for training
                    for x, y, z in self.train_data:  # get data batches
                        loss, accuracy = self.run_train_iter(x=x, y=y, z=z, model='enn')  # take a training iter step
                        enn_current_epoch_losses["train_loss"].append(loss)  # add current iter loss to the train loss list
                        enn_current_epoch_losses["train_acc"].append(accuracy)  # add current iter acc to the train acc list
                        pbar_train.update(1)
                        pbar_train.set_description("loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))     
                            
                with tqdm.tqdm(total=len(self.val_data)) as pbar_val:  # create a progress bar for validation
                    for x, y, z in self.val_data:  # get data batches
                        loss, accuracy = self.run_evaluation_iter(x=x, y=y, z=z, model='enn')  # run a validation iter
                        enn_current_epoch_losses["val_loss"].append(loss)  # add current iter loss to val loss list.
                        enn_current_epoch_losses["val_acc"].append(accuracy)  # add current iter acc to val acc lst.
                        pbar_val.update(1)  # add 1 step to the progress bar
                        pbar_val.set_description("loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))
    
                val_mean_accuracy = np.mean(enn_current_epoch_losses['val_acc'])
                if val_mean_accuracy > self.enn_best_val_model_acc:  # if current epoch's mean val acc is greater than the saved best val acc then
                    self.enn_best_val_model_acc = val_mean_accuracy  # set the best val model acc to be current epoch's val accuracy
                    self.enn_best_val_model_idx = epoch_idx  # set the experiment-wise best val idx to be the current epoch's idx
    
                for key, value in enn_current_epoch_losses.items():
                    enn_total_losses[key].append(np.mean(value))  # get mean of all metrics of current epoch metrics dict, to get them ready for storage and output on the terminal.
    
                out_string = "_".join(
                    ["{}_{:.4f}".format(key, np.mean(value)) for key, value in enn_current_epoch_losses.items()])
                # create a string to use to report our epoch metrics
                epoch_elapsed_time = time.time() - epoch_start_time  # calculate time taken for epoch
                epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
                print("Epoch {} (enn):".format(epoch_idx), out_string, "epoch time", epoch_elapsed_time, "seconds")
    
                self.save_model(model_save_name="train_enn_model", model_idx=epoch_idx,
                                best_validation_model_idx=self.enn_best_val_model_idx,
                                best_validation_model_acc=self.enn_best_val_model_acc)
                self.save_model(model_save_name="train_enn_model", model_idx='latest',
                                best_validation_model_idx=self.enn_best_val_model_idx,
                                best_validation_model_acc=self.enn_best_val_model_acc)
                
            print("best epoch: {}, best val acc: {}".format(self.enn_best_val_model_idx, self.enn_best_val_model_acc))

        # train pnn model
        print("Starting to train pnn")
        pnn_total_losses = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []} 
        if self.enn is not None:
            self.load_model(model_save_name="train_enn_model", model_idx=self.enn_best_val_model_idx)
        for i, epoch_idx in enumerate(range(self.num_epochs)):
            epoch_start_time = time.time()
            pnn_current_epoch_losses = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}
            self.current_epoch = epoch_idx
            
            with tqdm.tqdm(total=len(self.train_data)) as pbar_train:  # create a progress bar for training
                for x, y, z in self.train_data:  # get data batches
                    loss, accuracy = self.run_train_iter(x=x, y=y, z=z, model='pnn')  # take a training iter step
                    pnn_current_epoch_losses["train_loss"].append(loss)  # add current iter loss to the train loss list
                    pnn_current_epoch_losses["train_acc"].append(accuracy)  # add current iter acc to the train acc list
                    pbar_train.update(1)
                    pbar_train.set_description("loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy)) 
                    
            with tqdm.tqdm(total=len(self.val_data)) as pbar_val:  # create a progress bar for validation
                for x, y, z in self.val_data:  # get data batches
                    loss, accuracy = self.run_evaluation_iter(x=x, y=y, z=z, model='pnn')  # run a validation iter
                    pnn_current_epoch_losses["val_loss"].append(loss)  # add current iter loss to val loss list.
                    pnn_current_epoch_losses["val_acc"].append(accuracy)  # add current iter acc to val acc lst.
                    pbar_val.update(1)  # add 1 step to the progress bar
                    pbar_val.set_description("loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))
            
            val_mean_accuracy = np.mean(pnn_current_epoch_losses['val_acc'])
            if val_mean_accuracy > self.pnn_best_val_model_acc:  # if current epoch's mean val acc is greater than the saved best val acc then
                self.pnn_best_val_model_acc = val_mean_accuracy  # set the best val model acc to be current epoch's val accuracy
                self.pnn_best_val_model_idx = epoch_idx  # set the experiment-wise best val idx to be the current epoch's idx
            
            for key, value in pnn_current_epoch_losses.items():
                pnn_total_losses[key].append(np.mean(value))  # get mean of all metrics of current epoch metrics dict, to get them ready for storage and output on the terminal.

            
            out_string = "_".join(
                ["{}_{:.4f}".format(key, np.mean(value)) for key, value in pnn_current_epoch_losses.items()])
            # create a string to use to report our epoch metrics
            epoch_elapsed_time = time.time() - epoch_start_time  # calculate time taken for epoch
            epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
            print("Epoch {} (pnn):".format(epoch_idx), out_string, "epoch time", epoch_elapsed_time, "seconds")
           
            self.save_model(model_save_name="train_model", model_idx=epoch_idx,
                            best_validation_model_idx=self.pnn_best_val_model_idx,
                            best_validation_model_acc=self.pnn_best_val_model_acc)
            self.save_model(model_save_name="train_model", model_idx='latest',
                            best_validation_model_idx=self.pnn_best_val_model_idx,
                            best_validation_model_acc=self.pnn_best_val_model_acc)
            
        print("best epoch: {}, best val acc: {}".format(self.pnn_best_val_model_idx, self.pnn_best_val_model_acc))
            
        print("Generating test set evaluation metrics")
        self.load_model(model_save_name="train_model", model_idx=self.pnn_best_val_model_idx)
        enn_current_losses = {"test_acc": [], "test_loss": []}  # initialize a statistics dict
        pnn_current_losses = {"test_acc": [], "test_loss": []}  # initialize a statistics dict
        start_time = time.time()
        if self.enn is not None:
            with tqdm.tqdm(total=len(self.test_data)) as pbar_test:  # ini a progress bar
                for x, y, z in self.test_data:  # sample batch
                    loss, accuracy = self.run_evaluation_iter(x=x,
                                                              y=y,
                                                              z=z,
                                                              model='enn')  # compute loss and accuracy by running an evaluation step
                    enn_current_losses["test_loss"].append(loss)  # save test loss
                    enn_current_losses["test_acc"].append(accuracy)  # save test accuracy
                    pbar_test.update(1)  # update progress bar status
                    pbar_test.set_description(
                        "loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))  # update progress bar string output

        with tqdm.tqdm(total=len(self.test_data)) as pbar_test:  # ini a progress bar
            for x, y, z in self.test_data:  # sample batch
                loss, accuracy = self.run_evaluation_iter(x=x,
                                                          y=y,
                                                          z=z,
                                                          model='pnn')  # compute loss and accuracy by running an evaluation step
                pnn_current_losses["test_loss"].append(loss)  # save test loss
                pnn_current_losses["test_acc"].append(accuracy)  # save test accuracy
                pbar_test.update(1)  # update progress bar status
                pbar_test.set_description(
                    "loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))  # update progress bar string output
        
        test_elapsed_time = time.time() - start_time  # calculate time taken for epoch
        test_elapsed_time = "{:.4f}".format(test_elapsed_time)
        
        if self.enn is not None:
            out_string = "_".join(
                    ["{}_{:.4f}".format(key, np.mean(value)) for key, value in enn_current_losses.items()])
            # create a string to use to report our epoch metrics
            print("Test (enn):", out_string, "test time", test_elapsed_time, "seconds")
        out_string = "_".join(
                ["{}_{:.4f}".format(key, np.mean(value)) for key, value in pnn_current_losses.items()])
        print("Test (pnn):", out_string, "test time", test_elapsed_time, "seconds")
        
        if self.enn is not None:
            return enn_total_losses, pnn_total_losses
        else:
            return pnn_total_losses
        
        
        