# This is an example of how you would use this library
1. Create a custom dataset object by inheriting from `torch.utils.data.Dataset`.
1. Specify if you want to perform K-Fold Cross Validation. (!!!IN PROGRESS!!!)
1. Specify if you want to reproduce the experiment. (!!!IN PROGRESS!!!)
1. You must create an experiment file in YAML specifying the format of the experiment

    For the experiment file, you must specify the following:
    1. The name of the experiment
    1. Learning Rate
    1. Number of Epochs
    1. The loss function to be used:
        1. MSE -> "mse"
        1. Cross Entropy -> "ce"
        1. Binary Cross Entropy -> "bce"

1. Create the ML model in the models folder.
1. Make a trainer object
1. Train the model
1. Test the model