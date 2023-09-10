import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

PLOT_TIMEOUT = 0.25

def plot_train_losses(train_losses, title='Training Loss Curve'):
    """
    Plot training losses over epochs.

    Args:
        train_losses (list): List of training losses over epochs.
        title (str, optional): Title for the plot. Default is 'Training Loss Curve'.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show(block=False)
    plt.pause(PLOT_TIMEOUT)

class Trainer:
    def __init__(self, model, dataloader, experiment):
        self.model = model
        self.dataloader = dataloader
        self.num_epochs = experiment.num_epochs
        self.learning_rate = experiment.learning_rate

        if experiment.loss == "ce":
            print("Using Cross Entropy Loss")
            self.criterion = nn.CrossEntropyLoss()
        elif experiment.loss == "mse":
            print("Using Mean Squared Error Loss")
            self.criterion = nn.MSELoss()
        elif experiment.loss == "bce":
            print("Using Binary Cross Entropy Loss")
            self.criterion = nn.BCELoss()
        else:
            print("No loss function specified in experiment file. Using Mean Squared Error Loss.")
            self.criterion = nn.MSELoss()

        if experiment.optimizer == "sgd":
            print("Using SGD optimizer")
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        elif experiment.optimizer == "adam":
            print("Using Adam optimizer")
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif experiment.optimizer == "adagrad":
            print("Using Adagrad optimizer")
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        elif experiment.optimizer == "rmsprop":
            print("Using RMSProp optimizer")
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        else:
            print("No optimizer specified in experiment file. Using Adam.")
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        loss_per_epoch = []

        for epoch in range(self.num_epochs):
            running_loss = 0.0

            for inputs, labels in self.dataloader:
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # Print the average loss for this epoch
            average_loss = running_loss / len(self.dataloader)
            loss_per_epoch.append(average_loss)

            print(f"Epoch [{epoch + 1}/{self.num_epochs}] Train Loss: {average_loss:.4f} Learning Rate: {self.learning_rate:.4f}")
            plot_train_losses(loss_per_epoch)

        print("Training finished")