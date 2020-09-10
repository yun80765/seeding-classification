import torch
import torch.nn as nn
from models import VGG11
from dataset import SeedlingDataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import copy
import matplotlib.pyplot as plt


DATASET_ROOT = './'
use_gpu = torch.cuda.is_available()

def train():
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_set = SeedlingDataset(Path(DATASET_ROOT).joinpath('train'),data_transform)
    data_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True, num_workers=1)

    model = VGG11(num_classes=train_set.num_classes)
    if(use_gpu): model = model.cuda()
    model.train()

    best_model_params = copy.deepcopy(model.state_dict())
    best_acc=0.0
    num_epochs=100
    loss_values=[]
    acc_values=[]
    criterion = nn.MultiMarginLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.007)
    dataset_size = len(train_set)


    for epoch in range(num_epochs):
        print(f'Epoch: {epoch + 1}/{num_epochs}')
        print('-' * len(f'Epoch: {epoch + 1}/{num_epochs}'))

        training_loss = 0.0
        training_corrects = 0
        for i,(inputs,labels) in enumerate(data_loader):
            if(use_gpu):
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.data * inputs.size(0)
            training_corrects += torch.sum(preds == labels.data)

        training_loss = float(training_loss) / (dataset_size )
        training_acc = float(training_corrects) / (dataset_size)

        loss_values.append(training_loss)
        acc_values.append(training_acc)
        print(f'Train loss: {training_loss:.4f}\taccuracy: {training_acc:.4f}\n')

        if training_acc > best_acc:
            best_acc = training_acc
            best_model_params = copy.deepcopy(model.state_dict())

##### draw image
    plt.plot(loss_values)
    plt.title('loss curve')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('./images/svm_loss_curve.png')
    plt.show()

    plt.plot(acc_values)
    plt.title('accuracy curve')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.savefig('./images/svm_acc_curve.png')
    plt.show()


    model.load_state_dict(best_model_params)
    torch.save(model, f'model-{best_acc:.02f}-best_train_acc.pth')



if __name__ == '__main__':
    train()