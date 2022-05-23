import torch 
import torch.nn as nn 
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import numpy as np 
import os 
import gc
import argparse
from tqdm import tqdm 
from sklearn.metrics import accuracy_score
from cnn import CNN

torch.cuda.empty_cache()
gc.collect()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setseed(seed = 1024):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def evaluate(model, testloader):

    model.eval()
    output = []
    labels = []
    for image, label in tqdm(testloader):
        image, label = Variable(image), Variable(label)
        pred = model(image)
        pred = torch.argmax(pred, dim = 1)
        output.extend(pred.data.numpy())
        labels.extend(label.cpu().data.numpy())
        #print(pred)
        #print(label)
    #print(labels)
    #print(output)
    return accuracy_score(labels, output)

def train(model, trainloader, testloader, args):

    opt = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
    criteria = nn.CrossEntropyLoss()
    
    model.train()
    best_acc = 0 # to save model
    for epoch in range(args.epoch):
        # acc = evaluate(model, testloader)
        for idx, (image, label) in enumerate(tqdm(trainloader)):
            image, label = Variable(image), Variable(label)
            # print(image.size())
            output = model(image)
            loss = criteria(output, label)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if idx % args.print_ == 0:
                print("epoch: {}, loss: {}".format(epoch, loss))
        train_acc = evaluate(model, trainloader)
        test_acc = evaluate(model, testloader)
        print("train_acc: {}, test_acc: {} ".format(train_acc, test_acc))
        if test_acc > best_acc:
            best_acc = test_acc 
            torch.save(model, os.path.join(args.save_path, 'model.p'))
    print("best_acc: ", best_acc)


if __name__ == '__main__':
    
    setseed()
    
    train_mnist = datasets.MNIST(root = './data/', train = True, download=True, transform=transforms.ToTensor())
    test_mnist = datasets.MNIST(root = './data/', train = False, download = True, transform=transforms.ToTensor())

    parser = argparse.ArgumentParser(description = "The setting of CNN for MNIST")
    parser.add_argument('--save_path', default='model_save', type = str, help='path to save model')
    parser.add_argument('--lr', default=1e-3, type = float, help= 'learning rate')
    parser.add_argument('--batchsize', default=32, type = int, help= 'batch size')
    parser.add_argument('--epoch', default = 10, type = int, help = 'epoch num')
    parser.add_argument('--weight_decay', default=1e-6, type = float, help='l2 reg')
    parser.add_argument('--print_', default=500, type = int, help='interval num to control info')
    
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.save_path)):
        os.makedirs(os.path.join(args.save_path))
    
    with open(os.path.join(args.save_path, 'args.txt'), 'w') as f:
        f.write("\n".join([str(k) + ' = ' + str(v) for k,v in sorted(vars(args).items(), key=lambda x: x[0])]))

    trainloader = DataLoader(train_mnist, batch_size=args.batchsize, shuffle = True)
    testloader = DataLoader(test_mnist, batch_size=args.batchsize, shuffle=True)
    # train
    model = CNN()
    train(model, trainloader, testloader, args)
