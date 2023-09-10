from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class CustomMNISTDataset(Dataset):
    def __init__(self,
                 train=True,
                 data_dir="test",
                 transform=None):

        if transform is None:
            #By default, we will make a tensor out of the data
            transform = transforms.Compose([
                    transforms.ToTensor()
            ])

        self.data = datasets.MNIST(data_dir,
                                   train=train,
                                   transform=transform,
                                   download=True)
        self.data_dir = data_dir
        self.transform = transform

        type_str = "Train" if train else "Test"
        print(f"{type_str} Dataset made")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return image, label

class CustomDataset(Dataset):
    def __init__(self,
                 train=True,
                 data_dir=None,
                 transform=None):
        """
        This is a custom dataset that inherits from the torch dataset. If you want to make your own, inherit from this.
        """

        type_str = "Train" if train else "Test"
        print(f"{type_str} Dataset made")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return image, label

class CustomDataLoader(DataLoader):
    def __init__(self,
                 dataset=None,
                 train=True,
                 batch_size=16,
                 shuffle=True,
                 sampler=None,
                 batch_sampler=None,
                 num_workers=0,
                 collate_fn=None,
                 pin_memory=False,
                 drop_last=False,
                 timeout=0,
                 worker_init_fn=None,
                 multiprocessing_context=None,
                 generator=None, *,
                 prefetch_factor=None,
                 persistent_workers=False,
                 pin_memory_device=''):
        """
        This is a custom dataloader that inherits from the torch dataloader.
        """

        if dataset is None:
            print("Error! No dataset provided!")
            dataset = CustomMNISTDataset(train=train) #This is just for example

        super().__init__(dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         sampler=sampler,
                         batch_sampler=batch_sampler,
                         num_workers=num_workers,
                         collate_fn=collate_fn,
                         pin_memory=pin_memory,
                         drop_last=drop_last,
                         timeout=timeout,
                         worker_init_fn=worker_init_fn,
                         multiprocessing_context=multiprocessing_context,
                         generator=generator,
                         prefetch_factor=prefetch_factor,
                         persistent_workers=persistent_workers,
                         pin_memory_device=pin_memory_device)