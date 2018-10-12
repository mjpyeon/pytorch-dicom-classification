import numpy as np
import os
import glob
import torch
import torchvision
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

def extract_label(fname):
    return fname.split('__')[-1].split('.')[-3]

def multi_label(fnames):
    labeled = []
    with open('./labels.csv', 'r') as f:
        label_table = f.readlines()
    label_table = [s.replace('\n', '') for s in label_table]
    label_dict = {l:i for i, l in enumerate(label_table)}

    for f in fnames:
        labeled.append(label_dict[extract_label(f)])
    return np.array(labeled), label_table

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
class Normalize:
    def __call__(self, sample):
        image, label = sample
        image[:, 0] = (image[:, 0]-0.485)/0.229
        image[:, 1] = (image[:, 1]-0.456)/0.224
        image[:, 2] = (image[:, 2]-0.406)/0.225
        return (image, label)


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
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
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
    return model, history, best_acc

def save_history(fname, history):
    nb_epochs = len(history['train_loss'])
    with open(fname, 'w+') as f:
        f.write('epoch train_loss train_acc val_loss val_acc\n')
        for i in range(nb_epochs):
            f.write('%d %.4f %.4f %.4f %.4f\n'%(i, history['train_loss'][i],
                                                history['train_acc'][i],
                                                history['val_loss'][i],
                                                history['val_acc'][i]))



def train(architecture, output_dim, k, src, alloc_label, num_labels=2, lr=1e-3, betas=(0.9, 0.999), weight_decay=0, nb_epochs=25, batch_size=32, start_fold=0, end_fold=None):
    """
    k: "k"-fold
    src: k src lists
    alloc_label: fct to alloc labels
    define a dataset and the loader
    load a densenet pretrained using ImageNet
    train the network
    save the model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for curr_fold in range(start_fold, end_fold):
        train_src = []
        for i in range(k):
            if i != curr_fold:
                train_src.append(src[i])
        test_src = src[curr_fold]
        if architecture == "inception_v3":
            shape = (299, 299)
            output_dim = 2048
        else:
            shape = (256, 256)
        if architecture == 'resnet50' or architecture == 'resnet101' or architecture == 'resnet152':
            output_dim = 8192
        elif architecture == 'resnet18' or architecture == 'resnet34':
            output_dim = 2048
        train_dataset = EarDataset(binary_dir=train_src,
                                         alloc_label=alloc_label,
                                         transforms=transforms.Compose([Rescale(shape), ToTensor(), Normalize()]))
        test_dataset = EarDataset(binary_dir=test_src,
                                  alloc_label = alloc_label,
                                         transforms=transforms.Compose([Rescale(shape), ToTensor(), Normalize()]))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        dataloaders = {'train':train_loader, 'val':test_loader}
        dataset_sizes = {'train':len(train_dataset), 'val':len(test_dataset)}
        model_name = "resnet18"
        network = models.resnet18(pretrained=True).to(device)
        _global =  {"network":network, "models":models, "device":device, "model_name":model_name}
        exec("network = models.%s(pretrained=True).to(device)\nmodel_name=\'%s\'"%(architecture, architecture),_global)
        network = _global['network']
        model_name = _global['model_name']
        print(model_name,"is successfully loaded")
        #num_ftrs = network.fc.in_features
        #network.fc = nn.Linear(num_ftrs, num_labels).cuda()
        network.fc = nn.Linear(output_dim, num_labels).to(device)
        class_names = train_dataset.class_names
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(network.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        trained_model, curr_history, curr_best = train_model(network, criterion, optimizer, None, dataloaders, dataset_sizes, class_names, device, num_epochs=nb_epochs)
        save_history("%s_%.4facc_%dth_fold_lr-%.5f_beta1-%.2f_beta2-%.3f.csv"%(architecture, curr_best, curr_fold, lr, betas[0], betas[1]), curr_history)
        torch.save(trained_model, "%s_%.4facc_%dth-fold_lr-%.5f_beta1-%.2f_beta2-%.3f.pt"%(architecture, curr_best, curr_fold, lr, betas[0], betas[1]))

