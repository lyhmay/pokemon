import torch
import torchvision
from torch import optim,nn
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from pokemon import Pokemon
from resnet import ResNet18

batchsz = 32
lr =1e-3
epoches = 10

device = torch.device('cuda')
torch.manual_seed(1234)

train_db =Pokemon('pokemon',224,mode='train')
val_db =Pokemon('pokemon',224,mode='train')
test_db =Pokemon('pokemon',224,mode='train')

train_loader =DataLoader(train_db,batch_size = batchsz, shuffle =True,num_workers= 4)
val_loader =DataLoader(val_db,batch_size = batchsz,num_workers= 2)
test_loader =DataLoader(test_db,batch_size = batchsz,num_workers= 2)



def main():
    pass


if __name__ == '__main__':
    main()