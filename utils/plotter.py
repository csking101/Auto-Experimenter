import matplotlib.pyplot as plt

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
    plt.show()

def plot_losses(train_losses, test_losses, title='Loss Curves'):
    """
    Plot training and testing losses over epochs.

    Args:
        train_losses (list): List of training losses over epochs.
        test_losses (list): List of testing losses over epochs.
        title (str, optional): Title for the plot. Default is 'Loss Curves'.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy(accuracies, title='Accuracy Curve'):
    """
    Plot accuracy over epochs.

    Args:
        accuracies (list): List of accuracies over epochs.
        title (str, optional): Title for the plot. Default is 'Accuracy Curve'.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(accuracies, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Usage:
# Assuming you have lists of training and testing losses and accuracies
# train_losses = [0.1, 0.05, 0.03, ...]
# test_losses = [0.2, 0.1, 0.08, ...]
# accuracies = [90.0, 92.5, 94.0, ...]

# Plot training and testing losses
# plot_losses(train_losses, test_losses)

# Plot accuracy
# plot_accuracy(accuracies)
