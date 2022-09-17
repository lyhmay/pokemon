import torch
import torchvision
import time
from torch import optim,nn
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from pokemon import Pokemon
#from resnet import ResNet18

from torchvision.models import resnet18
from utils import Flatten

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


def evaluate(model , loader):
    correct = 0
    total = len(loader.dataset)

    for x,y in loader:
        x,y = x.to(device),y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred,y).sum().float().item()
    return correct/total

def main():
    #model = ResNet18(5).to(device)
    trained_model = resnet18(pretrained= True)
    model = nn.Sequential(*list(trained_model.children())[:-1],  #[b, 512, 1, 1]
                          Flatten(),
                          nn.Linear(512,5)
    ).to(device)
    # x =torch.randn(2,3,224,224)
    # print(model(x).shape) #[2, 512, 1, 1]
    #time.sleep(30)
    optimzer = optim.Adam(model.parameters(), lr = lr)
    criteon = nn.CrossEntropyLoss()


    writer = SummaryWriter(comment = 'transfer_train')

    best_epoch,best_acc = 0,0
    global_step=0
    for epoch in range(epoches):
        for step,(x,y) in enumerate(train_loader):

            #x [b,3,224,224] , y [b]
            x,y= x.to(device),y.to(device)

            logits =model(x)
            loss =criteon(logits,y)

            optimzer.zero_grad()
            loss.backward()
            optimzer.step()
            writer.add_scalar('transfer_train_loss',loss, global_step)
            global_step+=1
        if epoch%2 == 0:
            val_acc =evaluate(model,val_loader)
            if val_acc > best_acc:
                best_epoch =epoch
                best_acc   = val_acc
                #torch.save(model.state_dict(),'./ckpt/epoch'+str(epoch)+'_'+'Accuracy'+str(val_acc)+'.model')
                torch.save(model.state_dict(),'./ckpt/transfer_best.model')
            writer.add_scalar('transfer_val_accuracy', val_acc, epoch)
    print('best_epoch:',epoch,'best_acc: ', val_acc)

    model.load_state_dict(torch.load('./ckpt/transfer_best.model'))
    print('loaded from ckpt')

    test_acc = evaluate(model, test_loader)
    print('test_accuracy',test_acc)

if __name__ == '__main__':
    main()