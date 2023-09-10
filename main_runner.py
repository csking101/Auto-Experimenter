from dataset.loader import CustomDataLoader
from models.first_model import SimpleCNN
from training.trainer import CNNTrainer
from experiments.experimenter import Experiment
from testing.tester import MNISTAccuracyTester

train_data_loader = CustomDataLoader(train=True)
test_data_loader = CustomDataLoader(train=False)
print("Data Loader made")
model = SimpleCNN()
print("Model made")
experiment = Experiment(learning_rate=0.001,num_epochs=1)
print("Experiment made")
trainer = CNNTrainer(model,train_data_loader,experiment)
print("Trainer made")
trainer.train()
tester = MNISTAccuracyTester(model,test_data_loader)
print("Tester made")
tester.test_accuracy()