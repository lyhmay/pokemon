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
    model = ResNet18(5).to(device)
    optimzer = optim.Adam(model.parameters(), lr = lr)
    criteon = nn.CrossEntropyLoss()


    writer = SummaryWriter(comment = 'train')

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
            writer.add_scalar('train_loss',loss, global_step)
            global_step+=1
        if epoch%2 == 0:
            val_acc =evaluate(model,val_loader)
            if val_acc > best_acc:
                best_epoch =epoch
                best_acc   = val_acc
                #torch.save(model.state_dict(),'./ckpt/epoch'+str(epoch)+'_'+'Accuracy'+str(val_acc)+'.model')
                torch.save(model.state_dict(),'./ckpt/best.model')
            writer.add_scalar('val_accuracy', val_acc, epoch)
    print('best_epoch:',epoch,'best_acc: ', val_acc)

    model.load_state_dict(torch.load('./ckpt/best.model'))
    print('loaded from ckpt')

    test_acc = evaluate(model, test_loader)
    print('test_accuracy',test_acc)

if __name__ == '__main__':
    main()