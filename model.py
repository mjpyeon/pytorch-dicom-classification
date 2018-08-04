import numpy as np
import os
import glob
import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms, utils
from skimage.transform import resize
import collections
import time
import copy

def binary_label(fnames):
    """
    read file names and return their binary classes
    0: Normal
    1: Abnormal
    """
    labeled = []
    for f in fnames:
        if 'Normal' in f:
            labeled.append(0)
        else:
            labeled.append(1)
    return np.array(labeled), ["Normal", "Abnormal"]

class EarDataset(Dataset):
    def __init__(self, binary_dir, alloc_label, transforms=None):
        """
        binary_dir: directory where binary files (.npy files) exist
        allocate_label: a function to allocate labels
        transforms: ex. ToTensor
        load all file names
        allocate their classes
        """
        if not isinstance(binary_dir,str):
            self.fnames = []
            for curr_dir in binary_dir:
                self.fnames += glob.glob(os.path.join(curr_dir, "*"))
        else:
            self.fnames = glob.glob(os.path.join(binary_dir, "*"))
        self.labels, self.class_names = alloc_label(self.fnames)
        assert len(self.fnames) == len(self.labels), "Wrong labels"
        self.transforms = transforms

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        img = np.load(self.fnames[idx]).astype(np.float16)
        label = self.labels[idx]
        sample = (img, label)
        if self.transforms:
            try:
                sample = self.transforms(sample)
            except:
                for trs in self.transforms:
                    sample = trs(sample)
        return sample

class Rescale:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        rescaled = resize(sample[0], self.output_size, mode='constant')
        return (rescaled, sample[1])

class ToTensor:
    def __call__(self, sample):
        image, label = sample
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return (torch.FloatTensor(image), label)


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, class_names, device, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if scheduler:
                    scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            history["%s_loss"%(phase)].append(epoch_loss)
            history["%s_acc"%(phase)].append(epoch_acc)
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

def save_history(fname, history):
    nb_epochs = len(history['train_loss'])
    with open(fname, 'w+') as f:
        f.write('epoch train_loss train_acc val_loss val_acc\n')
        for i in range(nb_epochs):
            f.write('%d %.4f %.4f %.4f %.4f\n'%(i, history['train_loss'][i],
                                                history['train_acc'][i],
                                                history['val_loss'][i],
                                                history['val_acc'][i]))



def train(k, src, alloc_label, num_labels=2, lr=1e-3, betas=(0.9, 0.999), weight_decay=0, nb_epochs=25, batch_size=32):
    """
    k: "k"-fold
    src: k src lists
    alloc_label: fct to alloc labels
    define a dataset and the loader
    load a densenet pretrained using ImageNet
    train the network
    save the model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for curr_fold in range(k):
        train_src = []
        for i in range(k):
            if i != curr_fold:
                train_src.append(src[i])
        test_src = src[curr_fold]
        train_dataset = EarDataset(binary_dir=train_src,
                                         alloc_label=alloc_label,
                                         transforms=transforms.Compose([Rescale((256, 256)), ToTensor()]))
        test_dataset = EarDataset(binary_dir=test_src,
                                  alloc_label = alloc_label,
                                         transforms=transforms.Compose([Rescale((256, 256)), ToTensor()]))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        dataloaders = {'train':train_loader, 'val':test_loader}
        dataset_sizes = {'train':len(train_dataset), 'val':len(test_dataset)}
        network = models.resnet18(pretrained=True).to(device)
        #num_ftrs = network.fc.in_features
        #network.fc = nn.Linear(num_ftrs, num_labels).cuda()
        network.fc = nn.Linear(2048, num_labels).to(device)
        class_names = train_dataset.class_names
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(network.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        trained_model, curr_history = train_model(network, criterion, optimizer, None, dataloaders, dataset_sizes, class_names, device, num_epochs=nb_epochs)
        save_history("%dth_fold.csv"%(curr_fold), curr_history)
        torch.save(trained_model, "resnet161_%dth-fold.pt"%(curr_fold))

