import torch
import numpy as np
from tqdm import tqdm

from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F


# define the NN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # number of hidden nodes in each layer (512)
        hidden_1 = 512
        hidden_2 = 512
        # linear layer (784 -> hidden_1)
        self.fc1 = nn.Linear(28 * 28, hidden_1)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(hidden_2, 10)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.dropout(x)
        # add output layer
        x = self.fc3(x)
        return x


def special_bool(boolean: str):
    if type(boolean) == bool:
        return boolean
    else:
        if boolean == "True":
            return True
        else:
            return False


def bagging(x, y):
    x_y = np.concatenate([x, y], axis=1)
    bag = choices(x_y, k=len(x))
    x = [np.array(el[:-1]) for el in bag]
    y = [el[-1] for el in bag]
    return x, y


if __name__ == "__main__":
    import argparse
    import random
    import numpy as np
    from random import choices
    import os

    parser = argparse.ArgumentParser(description='TLBiLSTM network')
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU_id")
    parser.add_argument('--bagging', type=str, default=True, help="Bagging or Not")
    parser.add_argument('--ep', type=int, default=300, help="Number of Epochs")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--N', type=int, default=5, help="Number of Models")
    parser.add_argument('--T', type=float, default=0.5, help="Size of Dataset")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning Rate")
    parser.add_argument('--ds', type=str, default="mnist", help="Dataset")
    parser.add_argument('--keep_prob', type=float, default=0.6, help="Dropout Rate")
    parser.add_argument('--num_layers', type=int, default=1, help="Number of BiLSTM layer")
    parser.add_argument('--max_checkpoints', type=int, default=1)
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()
    print("\n" + "Arguments are: " + "\n")
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(12345)


    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = 20
    # percentage of training set to use as validation
    valid_size = 0.1
    # device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()

    # choose the training and test datasets
    if args.ds == 'mnist':
        train_data = datasets.MNIST(root='data', train=True,
                                    download=True, transform=transform)
    else:
        train_data = datasets.CIFAR10(root='data', train=True,
                                    download=True, transform=transform)

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = RandomSampler(train_data, replacement=True, num_samples=int(args.T*len(train_data)))
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders
    valid_loader = DataLoader(train_data, batch_size=batch_size,
                                               sampler=valid_sampler, num_workers=num_workers)

    train_loader = DataLoader(train_data, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=num_workers)

    model = Net()
    # model.cuda()

    # specify loss function (categorical cross-entropy)
    criterion = nn.CrossEntropyLoss()

    # specify optimizer (stochastic gradient descent) and learning rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # number of epochs to train the model
    n_epochs = args.ep

    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf  # set initial "min" to infinity

    for tr in range(args.N):
        for epoch in range(n_epochs):
            # monitor training loss
            train_loss = 0.0
            valid_loss = 0.0

            ###################
            # train the model #
            ###################
            model.train()  # prep model for training
            for data, target in tqdm(train_loader):
                torch.manual_seed(12345)
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the loss
                loss = criterion(output, target)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # update running training loss
                train_loss += loss.item() * data.size(0)

            ######################
            # validate the model #
            ######################
            model.eval()  # prep model for evaluation
            for data, target in valid_loader:
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the loss
                loss = criterion(output, target)
                # update running validation loss
                valid_loss += loss.item() * data.size(0)

            # print training/validation statistics
            # calculate average loss over an epoch
            train_loss = train_loss / len(train_loader.dataset)
            valid_loss = valid_loss / len(valid_loader.dataset)

            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch + 1,
                train_loss,
                valid_loss
            ))

            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min,
                    valid_loss))
                logdir = f'/home/charles/Desktop/deep_nlp_research/tmp/bagging/mlp/' \
                         f'bagging_{args.bagging}/N_{args.N}/{args.T}/{args.ds}/model_{tr}'
                if not os.path.exists(logdir):
                    os.mkdir(logdir)
                torch.save(model.state_dict(), os.path.join(logdir, 'model.pt'))
                valid_loss_min = valid_loss

