'''
This program is based on https://github.com/deeplearningturkiye/pratik-derin-ogrenme-uygulamalari/blob/master/PyTorch/rakam_tanima_CNN_MNIST.py

'''

'''
Dataset: MNIST (http://yann.lecun.com/exdb/mnist/) 
Algorithm: Convolutional Neural Networks
'''

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter

# arguments passed from the terminal
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--mode', type=str, default='train', 
                    help='train or test the model.')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model_path', type=str, default='models/mnist_cnn.pth', 
                    help='The path where the trained model is saved.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available() # make sure CUDA is avaialble

# generating a random seed
torch.manual_seed(args.seed)
if args.cuda: torch.cuda.manual_seed(args.seed)


# importing the MNIST dataset
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

# Convolutional Neural Network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # the model building blocks are defined below
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # Input channel: 1, Output channel: 10, Filter size: 5x5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # Input channel: 10, Output channel: 20, Filter size: 5x5

        # randomly drop out 50% of the neurons
        self.conv2_drop = nn.Dropout2d() # The default dropout rate is 50%

        self.fc1 = nn.Linear(320, 50) # number of input neurons: 320, number of output neurons: 50
        # 50 neurons fully connected layer.

        self.fc2 = nn.Linear(50, 10) # Number of input neurons: 50, Number of output neurons: 10
        # 10 neurons to represent our 10 classes.

    # now we can build the model based on the building blocks defined above
    def forward(self, x):
        # input(x) size: [1, 28, 28] x (batch_size) Channel size: 1, Image size: 28x28

        # we pass the input through the "conv1" layer we defined above,
        # then we add a MaxPooling layer
        # we then pass it through a ReLu activation layer:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # output size: [10, 12, 12]

        # then we add the Dropout layer that we defined above
        # then we apply a MaxPooling layer,
        # finally we pass it through a ReLu activation layer
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # output size: [10, 12, 12]

        x = x.view(-1, 320) 
        # this flattens x into a vector
        # 20 x 4 x 4 = 320:
        # output size: [320]

        # we reduce the dimension of x to 50 by passing it through fc1
        # we then pass our output through our ReLu activation layer:
        x = F.relu(self.fc1(x))
        # output size: [50]


        x = F.dropout(x, training=self.training)

        # we reduce the dimension of x to 10 by passing it through fc2
        x = self.fc2(x)
        # output size: [10]
        # we have 10 outputs to represent 10 classes in our dataset

        # finally we use the Softmax function to transform x into a posterior distribution
        return F.log_softmax(x,-1)

# create an instance of our CNN      
model = Net()
# move the model to the GPU
if args.cuda: model.cuda()

# the Stochastic Gradient Descent optimizer that we will use to train our network (i.e., minimise the loss [defined below])
# https://en.wikipedia.org/wiki/Stochastic_gradient_descent
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# train the network
def train():
    
    # log the training information so we can later visualise it using tensorboard
    tb = SummaryWriter('./log_pytorch')

    # put the model into training mode
    model.train()
    # start the training, which will run for several epochs (https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/)
    for epoch in range(1, args.epochs + 1):
        # keep track of the number of correctly classified samples during each epoch
        correct = 0
        # create our function to train the model
        # divide the dataset into batches (https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/)
        for batch_idx, (data, target) in enumerate(train_loader):
            # move the data to the GPU
            if args.cuda: data, target = data.cuda(), target.cuda()
            # convert the data to PyTorch variables (Tensor)
            #data, target = Variable(data), Variable(target)
            # initialise the gradients to zero
            optimizer.zero_grad()
            # feed the input data into the CNN model and get the output
            output = model(data)
            # calculate the error by comparing the result that should be obtained with the output produced by our model
            # here we use the negative log likelihood loss (https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html)
            loss = F.nll_loss(output, target)
            # once we have measure the error, we apply back-propagation to calculate the gradients needed to update the model parameters
            # https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            loss.backward()
            # update the model parameters (weights) using SGD
            optimizer.step()
            # the predicted class is the one with the maximum probability in the posterior probability vector (output of softmax)
            pred = output.data.max(1)[1]
            # count the number of times the prediction made by our model is correct (i.e., equal to the target)
            correct += pred.eq(target.data.view_as(pred)).cpu().sum() # M
    
            # print the current loss
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item())) 
                    
        # add train loss to tensorboard
        tb.add_scalar("epoch loss", loss.item(), epoch)
        # add accuracy to tensorboard
        tb.add_scalar("epoch accuracy", 100. * correct / len(train_loader.dataset), epoch)
        # add weight histogram to tensorboard
        for name, weight in model.named_parameters():
            tb.add_histogram(name, weight, epoch)
            tb.add_histogram(f'{name}.grad',weight.grad, epoch)
            
    # save the model to a .pth file
    print('Saving CNN to %s' % args.model_path)
    torch.save(model.state_dict(), args.model_path)
    # add graph to tensorboard
    tb.add_graph(model, (data,))
        

# evaluate the model performance (accuracy) on the test data
def test():
    
    # load the model we trained and saved in args.model_path
    model.load_state_dict(torch.load(args.model_path,weights_only=True))
    
    # put model into test mode
    model.eval()
    # initialise the variables used to track the loss and the correct predictions made by the model
    test_loss = 0
    correct = 0
    # fetch the test data
    for data, target in test_loader:
        if args.cuda: data, target = data.cuda(), target.cuda()
        # no need to compute gradients now, we're only "using" the CNN, we're not training it
        with torch.no_grad():
            # convert the data to PyTorch variables (Tensor)
            # data, target = Variable(data, volatile=True), Variable(target)
            # feed the input data into the CNN model and get the output
            output = model(data)
            # calculate the loss for this batch of data points and add it to the total
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # the predicted class is the one with the maximum probability in the posterior probability vector (output of softmax)
            pred = output.data.max(1)[1]
            # count the number of times the prediction made by our model is correct (i.e., equal to the target)
            correct += pred.eq(target.data.view_as(pred)).cpu().sum() # M

    # calculate the average loss
    test_loss /= len(test_loader.dataset)
    # print the average loss and the percentage of correct classifications made by our model (i.e., accuracy)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100*correct / len(test_loader.dataset)))

if __name__ == '__main__':

    if args.mode == 'train': train()
    elif args.mode == 'test': test() 