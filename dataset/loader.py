from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class CustomMNISTDataset(Dataset):
    #This is created to test out the module
    def __init__(self, data_dir="test", transform=None):
        transform = transforms.Compose([
                # you can add other transformations in this list
                transforms.ToTensor()
            ])
        self.data = datasets.MNIST(data_dir, train=True, transform=transform, download=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return image, label

class CustomDataLoader(DataLoader):
    #This is created to test out the module
    def __init__(self, data_dir="test", batch_size=16, num_workers=0, shuffle=True, transform=None):
        custom_dataset = CustomMNISTDataset(data_dir, transform=transform)
        super().__init__(custom_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
