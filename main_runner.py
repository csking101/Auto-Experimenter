from dataset.loader import CustomDataLoader
from models.first_model import SimpleCNN
from training.trainer import CNNTrainer
from experiments.experimenter import Experiment
from testing.tester import MNISTAccuracyTester

data_loader = CustomDataLoader()#Need to differentiate between train and test
print("Data Loader made")
model = SimpleCNN()
print("Model made")
experiment = Experiment(learning_rate=0.001,num_epochs=1)
print("Experiment made")
trainer = CNNTrainer(model,data_loader,experiment)
print("Trainer made")
trainer.train()
tester = MNISTAccuracyTester(model,data_loader)
print("Tester made")
tester.test_accuracy()