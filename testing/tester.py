import torch

class MNISTAccuracyTester:
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader

    def test_accuracy(self):
        self.model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"The accuracy of the model is {accuracy:.2f}")
        return accuracy
