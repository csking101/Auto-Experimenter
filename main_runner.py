from dataset.loader import CustomDataLoader
from models.model_list import get_model_global
from training.trainer import Trainer
from experiments.experimenter import make_experiment
from testing.tester import MNISTAccuracyTester


experiment = make_experiment("./experiments/test_experiment_file.yaml")
print("Experiment made")
train_data_loader = CustomDataLoader(train=True,experiment=experiment)
test_data_loader = CustomDataLoader(train=False,experiment=experiment)
print("Data Loader made")
model = get_model_global(experiment.model_name)
print("Model made")
trainer = Trainer(model,train_data_loader,experiment)
print("Trainer made")
trainer.train()
tester = MNISTAccuracyTester(model,test_data_loader)
print("Tester made")
tester.test_accuracy()