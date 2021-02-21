
from dataset import LineFollowerDataset
from sim import Action
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18
from torchvision import transforms
from tqdm import tqdm
import time
import copy

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            # for inputs, labels in dataloaders[phase]:
            for batch_id, sample_batched in enumerate(dataloaders[phase]):
                inputs = sample_batched[0].to(device)
                labels = sample_batched[1].to(device)
                #labels = labels.to(device)

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
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        #print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # TODO: Load the dataset
    # - You might want to use a different transform than what is already provided
    
    dataset = LineFollowerDataset(transform=transforms.ToTensor())
    
    # TODO: Prepare dataloaders
    # - Rnadomly split the dataset into the train validation dataset.
    # 	* Hint: Checkout torch.utils.data.random_split
    # - Prepare train validation dataloaders
    # ========================================
    train_size = int(len(dataset.data)*0.8)
    val_size = len(dataset.data)-train_size
    
    dataset_sizes = {'train': train_size, 'val': val_size}
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=True)

    dataloaders = {'train': train_loader, 'val': val_loader}
    
    # ========================================

    # TODO: Prepare model
    # - You might want to use a pretrained model like resnet18 and finetune it for your dataset
    # - See https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    # ========================================
    model = resnet18(pretrained=True)

    #for param in model.parameters():
        #param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model.fc.in_features
    

    model.fc = nn.Sequential(
                      nn.Linear(num_ftrs, 256), 
                      nn.ReLU(), 
                      nn.Linear(256, 128),
                      nn.ReLU(),
                      nn.Linear(128, 4),                                   
                      nn.LogSoftmax(dim=1))
                              
    model = model.to(device)
    
    print(model)

    

    # TODO: Prepare loss and optimizer
    # ========================================
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()
    #optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.fc.parameters())
    # Decay LR by a factor of 0.1 every 7 epochs
    # ========================================
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # ========================================

    # TODO: Train model
    # - You might want to print training (and validation) loss and accuracy every epoch
    # - You might want to save your trained model every epoch
    # ========================================
    # ========================================
    
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler, num_epochs=25)

    torch.save(model, 'extra_following_model.pth.')