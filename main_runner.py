from dataset.loader import CustomDataLoader
from models.first_model import SimpleCNN
from training.trainer import Trainer
from experiments.experimenter import make_experiment
from testing.tester import MNISTAccuracyTester



train_data_loader = CustomDataLoader(train=True)
test_data_loader = CustomDataLoader(train=False)
print("Data Loader made")
model = SimpleCNN()
print("Model made")
experiment = make_experiment("./experiments/test_experiment_file.yaml")
print("Experiment made")
trainer = Trainer(model,train_data_loader,experiment)
print("Trainer made")
trainer.train()
tester = MNISTAccuracyTester(model,test_data_loader)
print("Tester made")
tester.test_accuracy()