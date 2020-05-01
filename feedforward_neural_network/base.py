import logging
import os
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import torch.optim.lr_scheduler as lr
from torch.utils.data import Dataset

from utils.logger import get_logger


LOGGER = get_logger(name="neural_network_experimental.py", level=logging.INFO)


class FeedForwardNetwork:
    def __init__(
            self,
            network: torch.nn.Module,
            criterion: torch.nn.modules.loss,
            optimizer: Optimizer,
            data_type: torch.dtype = torch.float32,
            batch_size: int = 256,
            shuffle_training_examples: bool = False,
            scheduler: Optional[lr._LRScheduler] = None,
    ):
        self.network = network
        if network:
            self.network_initial_state_dict = network.state_dict()
        self.criterion = criterion
        self.optimizer = optimizer
        if optimizer:
            self.optimizer_initial_state_dict = optimizer.state_dict()
        self.data_type = data_type
        self.batch_size = batch_size
        self.shuffle_training_examples = shuffle_training_examples
        if scheduler:
            self.scheduler = scheduler
        else:
            self.scheduler = lr.StepLR(self.optimizer, step_size=1, gamma=0.99)
        self.scheduler_initial_state_dict = self.scheduler.state_dict()

        # placeholders until functions which assign them are called
        self.network_output = None
        self.training_average_loss = None
        self.validation_average_loss = None
        self.epochs_trained = 0
        self.maximum_epochs_allowed = None
        self.training_dataloader = None
        self.validation_dataloader = None
        self.learned_params = None

    @staticmethod
    def generate_training_validation_split(X: np.ndarray, y: Union[np.ndarray, np.array]):
        """Split a dataset into training and validation sets. Uses consistent random seeding so that the same split is
        generated given the same arguments over multiple calls. Arguments and returned objects are all numpy arrays.
        Args:
            X (numpy.ndarray): Represents a design matrix. Should not contain any nan values.
            y (numpy.ndarray/numpy.array): Represents ground truths. Should not contain any nan values.
        Returns:
            (np.ndarray, np.array/np.ndarray, np.ndarray, np.array/np.ndarray): The 0th and 2nd elements are the
                training and validation design matrices respectively. The 1st and 3rd elements are the training and
                validation targets respectively.
        """
        # an 80/20 split by default seems reasonable
        PORTION_ASSIGNED_TRAINING = 0.8
        num_data_points = X.shape[0]
        assert num_data_points == y.shape[0]
        # seed the RNG so that we get consistent results across multiple executions
        np.random.seed(1)
        training_indices = np.random.choice(
            range(X.shape[0]), size=int(PORTION_ASSIGNED_TRAINING * num_data_points), replace=False
        )
        validation_indices = np.setdiff1d(np.arange(X.shape[0]), training_indices)
        training_design_matrix = X[training_indices]
        training_targets_array = y[training_indices]
        validation_design_matrix = X[validation_indices]
        validation_targets_array = y[validation_indices]
        return training_design_matrix, training_targets_array, validation_design_matrix, validation_targets_array

    @staticmethod
    def generate_dataloader(
            design_matrix: Union[np.ndarray, torch.Tensor],
            targets_array: Union[np.array, torch.Tensor],
            data_type: torch.dtype = torch.float32,
            batch_size: int = 256,
            shuffle: bool = False,
            num_workers: int = 0,
    ):
        """
        Convert two np arrays or torch Tensors - design matrix and targets - into torch.utils.data.DataLoader instance.
        Args:
            design_matrix (numpy.ndarray or torch.Tensor): represents a design matrix
            targets_array (numpy.array/torch.Tensor): represents ground truth targets
            data_type (torch.dtype): the data type (eg uint8, float64) of the design matrix & target
            batch_size (int): the number of rows returned by the returned DataLoader generator per batch
            shuffle (bool): cycles through all batches randomly if True; else returns in order
            num_workers (int): number of parallel threads generating batches concurrently at run time
        Returns:
            torch.utils.data.DataLoader: a generator, each item of which is a tuple of the form
                (design_matrix_batch : torch.Tensor, label_array : torch.Tensor); see
                https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel and
                https://pytorch.org/docs/stable/data.html for further documentation of DataLoader class
            """
        dataloader_wrapper_args = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
        }
        assert type(design_matrix) in (torch.Tensor, np.ndarray)
        assert type(targets_array) in (torch.Tensor, np.ndarray, np.array)

        if type(design_matrix) is np.ndarray:
            design_matrix_as_torch_tensor = torch.from_numpy(design_matrix).type(data_type)
        else:
            design_matrix_as_torch_tensor = design_matrix.type(data_type)

        if type(targets_array) in (np.array, np.ndarray):
            targets_array_as_torch_tensor = torch.from_numpy(targets_array).type(data_type)
        else:
            targets_array_as_torch_tensor = targets_array.type(data_type)
        dataset = torch.utils.data.TensorDataset(design_matrix_as_torch_tensor, targets_array_as_torch_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, **dataloader_wrapper_args)
        return dataloader

    def initialize_dataloaders(
            self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, np.array]
    ):
        """Create torch Dataloaders from raw design_matrix/ground truth array for a training and validation set and
        assign to instance's training_dataloader and validation_dataloader instances, respectively. See documentation
        for generate_training_validation_split and generate_dataloader for how this splitting/creation is done.
        Args:
            X (np.ndarray): Design matrix; set of signals for points in dataset.
            y (np.ndarray/np.array): Array representing ground truths in data set.
        Returns:
            None: self.training_dataloader and self.validation_dataloader assigned upon success
        """
        training_design_matrix, training_targets_array, validation_design_matrix, validation_targets_array = self.generate_training_validation_split(
            X, y
        )
        training_dataloader_kwargs = {
            "design_matrix": training_design_matrix,
            "targets_array": training_targets_array,
            "data_type": self.data_type,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle_training_examples,
        }
        validation_dataloader_kwargs = {
            "design_matrix": validation_design_matrix,
            "targets_array": validation_targets_array,
            "data_type": self.data_type,
            "batch_size": self.batch_size,
            "shuffle": False,
        }
        self.training_dataloader = self.generate_dataloader(**training_dataloader_kwargs)
        self.validation_dataloader = self.generate_dataloader(**validation_dataloader_kwargs)

    def forward(self, input_tensor: torch.Tensor):
        """
        Propagate input_tensor through the network, assigning the result to self.output. Call backprop after forward
        to take a training step.
        Args:
            input_tensor (torch.Tensor): A matrix where each row represents a data point and each column represents
                a data field
        Returns:
            torch.Tensor: The output from forward propagation of input_tensor
        """
        self.network_output = self.network.forward(input_tensor.type(self.data_type))
        return self.network_output

    def backprop(self, targets: torch.Tensor):
        """
        Given targets, calculate loss and call self.optimizer to update network weights. backprop assumes forward
        was called previously and looks up the value of self.output to calculate loss.
        Args:
            targets (torch.Tensor): The ground truths for data points whose whose predicted probabilities are in
                self.output.
        Returns:
            None: network weights are updated on successful execution
        """
        assert self.network_output is not None
        resized_targets = targets.view(-1)
        self.optimizer.zero_grad()
        loss = self.criterion(self.network_output, resized_targets)
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

    def training_step(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor):
        """
        Take a training step (ie, forward propagation followed by backward propagation)
        Args:
            input_tensor (torch.Tensor): Design matrix (or subset of) to be forward propagated through network
            target_tensor (torch.Tensor): Ground truths for data points represented in input_tensor
        Returns:
            None (network weights updated on successful execution)
        """
        self.forward(input_tensor)
        self.backprop(target_tensor)

    # helper function called in training_loss, validation_loss, test_loss
    def calculate_average_loss(self, dataloader: torch.utils.data.DataLoader):
        """
        Args:
            dataloader (torch.utils.data.Dataset): Generic torch dataloader, although presumably this argument
                will be one of self.training_dataloader or self.validation_dataloader which are in turn instantiated
                by self.initialize_dataloaders
        Returns:
            float: Average loss (calculated by self.criterion) across all batches in dataloader
        """
        running_loss = 0
        for batch, targets in dataloader:
            prediction = self.forward(batch)
            resized_targets = targets.view(-1)
            loss = self.criterion(prediction, resized_targets)
            running_loss += float(loss)
        return running_loss / len(dataloader.dataset)

    def calculate_training_loss(self):
        """
        Call self.calculate_average_loss on self.training_dataloader and assign result to
        self.training_average_loss.
        """
        self.network.train()
        self.training_average_loss = self.calculate_average_loss(self.training_dataloader)

    def calculate_validation_loss(self):
        """
        Call self.calculate_average_loss on self.validation_dataloader and assign result to
        self.validation_average_loss.
        """
        self.network.train()
        self.validation_average_loss = self.calculate_average_loss(self.validation_dataloader)

    def train_epoch(self):
        """
        Train network for one full epoch. training_dataloader must be assigned a value first, presumably by the
        initialize_dataloaders method, as it is None by default upon initialization. See forward and backprop methods
        for mechanics of training.
        """
        for batch, targets in self.training_dataloader:
            self.training_step(batch, targets)
        self.calculate_training_loss()
        self.epochs_trained += 1
        LOGGER.info(
            "Training loss after {} epochs: {}".format(str(self.epochs_trained), str(self.training_average_loss))
        )

    def _train(self):
        """
        Train the network to fit the data in self.training_dataloader.
        Note the difference between this method ("_train") and train. This method contains all the machine learning
        logic -- forward prop, backprop, adjusting LR etc. train contains "software logic" -- creating dataloader
        objects, re-initializing instance attributes etc.
        Also can use self.epoches to set constrain for self.epoch_trained to manually break optimization
        Returns:
            None (updates network weights over course of training)
        """
        self.network.train()  # note that here we are calling torch.nn.Module train class method
        epochs_since_improvement = 0
        best_params = None
        self.calculate_validation_loss()
        best_validation_loss = self.validation_average_loss

        while epochs_since_improvement < 10:
            self.train_epoch()
            self.calculate_validation_loss()
            if self.validation_average_loss < best_validation_loss:
                epochs_since_improvement = 0
                best_validation_loss = self.validation_average_loss
                best_params = self.network.state_dict()
            else:
                epochs_since_improvement += 1
            LOGGER.info("Epochs since improvement in validation_loss: {} \n".format(epochs_since_improvement))
            if self.maximum_epochs_allowed is not None and self.epochs_trained >= self.maximum_epochs_allowed:
                break
        LOGGER.info("Training complete after {} epochs \n".format(self.epochs_trained))
        LOGGER.info("Best training loss achieved: {} \n".format(self.training_average_loss))
        LOGGER.info("Best validation loss achieved: {}".format(self.validation_average_loss))
        self.learned_params = best_params
        self.network.load_state_dict(best_params)

    def fit(
            self,
            X: Optional[np.ndarray],
            y: Optional[Union[np.ndarray, np.array, pd.Series]],
    ):
        """
        Given dataset as well as all other attributes (criterion, optimizer, etc), train the network. Note the
        difference between this method and _train -- this method contain all "software logic" (creating dataloader
        objects, re-initializing instance attributes etc) and _train contains all "ML logic" (forward and backprop, etc)
        Note:
            if targets/design_matrix are left as None, training will be attempted on value previously assigned to
            self.training_dataloader. If not value has been assigned, execution will fail.
            this generic method should be overwritten by a superclass neural network algo.
        Args:
            X (np.ndarray, optional): training design matrix
            y (np.ndarray, optional): training ground truths
        Returns:
            None (modifies self.learned_params on successful execution)
        """
        self.epochs_trained = 0
        # if design_matrix/targets None assume dataloaders were initialized elsewhere
        if X is not None:
            if type(y) is pd.Series:
                self.initialize_dataloaders(X, np.array(y))
            else:
                assert type(y) in (np.ndarray, np.array)
                self.initialize_dataloaders(X, y)
        self.network.load_state_dict(self.network_initial_state_dict)
        self.optimizer.load_state_dict(self.optimizer_initial_state_dict)
        if self.scheduler:
            self.scheduler.load_state_dict(self.scheduler_initial_state_dict)
        self._train()

    def predict(self, X: np.ndarray):
        """Make predictions on given input_matrix, ie feed input matrix through net. Forward propagation without the
        corresponding backprop.
        Note:
            This method returns a np.ndarray. If a year argument is given, it will also store the result in
            self.learned_params[year]; otherwise it will simply return.
            Also, if no year parameter is given the network will use whatever set of weights are already loaded into
            the network rather than loading in self.learned_params[year]
            this generic method should be overwritten by a superclass neural network algo.
        Args:
            X (np.ndarray): a design matrix
        Returns:
            np.ndarray -- the predictions return by the network for the given learned parameters and design matrix
        """
        self.network.eval()
        input_tensor = torch.from_numpy(X)
        self.network.load_state_dict(self.learned_params)
        try:
            output = self.forward(input_tensor).numpy()
        except:
            output = self.forward(input_tensor.data).detach().numpy()
        return output





