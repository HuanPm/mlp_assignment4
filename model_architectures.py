import torch
import torch.nn as nn
import torch.nn.functional as F


class ENN(nn.Module):
    def __init__(self, input_shape, num_output_classes, num_filters, num_layers, use_bias=False):
        """
        Initializes a fully connected network similar to the ones implemented previously in the MLP package.
        :param input_shape: The shape of the inputs going in to the network.
        :param num_output_classes: The number of outputs the network should have (for classification those would be the number of classes)
        :param num_filters: Number of filters used in every fcc layer.
        :param num_layers: Number of ENN layers (excluding dim reduction stages)
        :param use_bias: Whether our ENN layers will use a bias.
        """
        super(ENN, self).__init__()
        # set up class attributes useful in building the network and inference
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.num_output_classes = num_output_classes
        self.use_bias = use_bias
        self.num_layers = num_layers
        # initialize a module dict, which is effectively a dictionary that can collect layers and integrate them into pytorch
        self.layer_dict = nn.ModuleDict()
        # build the network
        self.build_module()
        
    def build_module(self):
        print("Building basic block of ENN using input shape", self.input_shape)
        x = torch.zeros((self.input_shape))

        out = x
        out = out.view(out.shape[0], -1)
        # flatten inputs to shape (b, -1) where -1 is the dim resulting from multiplying the
        # shapes of all dimensions after the 0th dim

        for i in range(self.num_layers):
            self.layer_dict['enn_{}'.format(i)] = nn.Linear(in_features=out.shape[1],  # initialize a enn layer
                                                            out_features=self.num_filters,
                                                            bias=self.use_bias)

            out = self.layer_dict['enn_{}'.format(i)](out)  # apply ith enn layer to the previous layers outputs
            out = F.relu(out)  # apply a ReLU on the outputs

        self.logits_linear_layer = nn.Linear(in_features=out.shape[1],  # initialize the prediction output linear layer
                                             out_features=self.num_output_classes,
                                             bias=self.use_bias)
        out = self.logits_linear_layer(out)  # apply the layer to the previous layer's outputs
        print("Block is built, output volume is", out.shape)
        return out

    def forward(self, x):
        """
        Forward prop data through the network and return the preds
        :param x: Input batch x a batch of shape batch number of samples, each of any dimensionality.
        :return: preds of shape (b, num_classes)
        """
        out = x
        out = out.view(out.shape[0], -1)
        # flatten inputs to shape (b, -1) where -1 is the dim resulting from multiplying the
        # shapes of all dimensions after the 0th dim

        for i in range(self.num_layers):
            if i%2 == 1:
                y = out
            
            out = self.layer_dict['enn_{}'.format(i)](out)  # apply ith enn layer to the previous layers outputs
            
            if i%2 == 0 and i > 0:
                out = out + y
            
            out = F.relu(out)  # apply a ReLU on the outputs

        out = self.logits_linear_layer(out)  # apply the layer to the previous layer's outputs
        out = F.sigmoid(out)
        return out

    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        for item in self.layer_dict.children():
            item.reset_parameters()

        self.logits_linear_layer.reset_parameters()
    
class PNN(nn.Module):
    def __init__(self, input_shape, num_output_classes, num_filters, num_layers, use_bias=False):
        """
        Initializes a fully connected network similar to the ones implemented previously in the MLP package.
        :param input_shape: The shape of the inputs going in to the network.
        :param num_output_classes: The number of outputs the network should have (for classification those would be the number of classes)
        :param num_filters: Number of filters used in every fcc layer.
        :param num_layers: Number of ENN layers (excluding dim reduction stages)
        :param use_bias: Whether our ENN layers will use a bias.
        """
        super(PNN, self).__init__()
        # set up class attributes useful in building the network and inference
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.num_output_classes = num_output_classes
        self.use_bias = use_bias
        self.num_layers = num_layers
        # initialize a module dict, which is effectively a dictionary that can collect layers and integrate them into pytorch
        self.layer_dict = nn.ModuleDict()
        # build the network
        self.build_module()
        
    def build_module(self):
        print("Building basic block of PNN using input shape", self.input_shape)
        x = torch.zeros((self.input_shape))

        out = x
        out = out.view(out.shape[0], -1)
        # flatten inputs to shape (b, -1) where -1 is the dim resulting from multiplying the
        # shapes of all dimensions after the 0th dim

        for i in range(self.num_layers):
            self.layer_dict['pnn_{}'.format(i)] = nn.Linear(in_features=out.shape[1],  # initialize a fcc layer
                                                            out_features=self.num_filters,
                                                            bias=self.use_bias)

            out = self.layer_dict['pnn_{}'.format(i)](out)  # apply ith pnn layer to the previous layers outputs
            out = F.relu(out)  # apply a ReLU on the outputs

        self.logits_linear_layer = nn.Linear(in_features=out.shape[1],  # initialize the prediction output linear layer
                                             out_features=self.num_output_classes,
                                             bias=self.use_bias)
        out = self.logits_linear_layer(out)  # apply the layer to the previous layer's outputs
        print("Block is built, output volume is", out.shape)
        return out

    def forward(self, x):
        """
        Forward prop data through the network and return the preds
        :param x: Input batch x a batch of shape batch number of samples, each of any dimensionality.
        :return: preds of shape (b, num_classes)
        """
        out = x
        out = out.view(out.shape[0], -1)
        # flatten inputs to shape (b, -1) where -1 is the dim resulting from multiplying the
        # shapes of all dimensions after the 0th dim
        
        for i in range(self.num_layers):
            if i%2 == 1:
                y = out
            
            out = self.layer_dict['pnn_{}'.format(i)](out)  # apply ith pnn layer to the previous layers outputs
            
            if i%2 == 0 and i > 0:
                out = out + y
            
            out = F.relu(out)  # apply a ReLU on the outputs

        out = self.logits_linear_layer(out)  # apply the layer to the previous layer's outputs
      
        return out

    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        for item in self.layer_dict.children():
            item.reset_parameters()

        self.logits_linear_layer.reset_parameters()
