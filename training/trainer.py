import torch
import torch.nn as nn
import torch.optim as optim

class CNNTrainer:
    def __init__(self, model, dataloader, experiment):
        self.model = model
        self.dataloader = dataloader
        self.num_epochs = experiment.num_epochs
        self.learning_rate = experiment.learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
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
            print(f"Epoch [{epoch + 1}/{self.num_epochs}] Loss: {average_loss:.4f}")

        print("Training finished")

# Usage:
# Assuming you have your dataset and model ready
# dataloader = ...
# model = SimpleCNN(num_classes)
# trainer = CNNTrainer(model, dataloader)
# trainer.train()