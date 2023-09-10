import yaml

class Experiment():
    #These will be the parameters for all of our experiments
    def __init__(self,name,learning_rate,num_epochs,loss,optimizer,batch_size,num_workers,shuffle):
        self.name = name
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

def make_experiment(file_name):
    """
    This function will make and return the experiment object, from the YAML file specified by file_name.
    """

    with open(file_name,'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        print(f"The file {file_name} has been loaded")
        print(f"The specifications are: {data}")

        experiment = Experiment(data['name'],
                                data['learning_rate'],
                                data['num_epochs'],
                                data['loss'],
                                data['optimizer'],
                                data['batch_size'],
                                data['num_workers'],
                                data['shuffle']
                                )

        return experiment