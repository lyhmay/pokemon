import torch
import os,glob
import  random,csv
import numpy as np

from torch.utils.data import Dataset,DataLoader
from torchvision import  transforms
from PIL import Image

from tensorboardX import SummaryWriter


class Pokemon(Dataset):
    def __init__(self, root, resize, mode):
        super(Pokemon, self).__init__()
        self.root =root
        self.resize =resize
        self.name2label = {}
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root,name)):
                continue
            self.name2label[name] =len(self.name2label.keys())
        print(self.name2label)
        self.images,self.labels=self.load_csv('images.csv')

        if mode=='train':
            self.images = self.images[0:int(0.6*len(self.images))]
            self.labels = self.labels[0:int(0.6*len(self.labels))]
        elif mode == 'val':
            self.images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.images))]
        elif mode == 'test':
            self.images = self.images[int(0.8* len(self.images)): len(self.images)]
            self.labels = self.labels[int(0.8* len(self.labels)): len(self.images)]


    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            # pokemon\\mewtwo\\00001.png
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
                # images+= glob.glob(os.path.join(self.root, name, '*.gif'))
            print(len(images), name)
            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='')as f:
                writer = csv.writer(f)
                for img in images:  # pokemon\\mewtwo\\00001.png
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    writer.writerow([img, label])
                print('written into csv file:', filename)

        images, labels = [], []
        with open(os.path.join(self.root,filename))as f:
            reader = csv.reader(f)
            for row in reader:
                img,label = row
                label =int(label)

                images.append(img)
                labels.append(label)
        assert len(images) == len(labels)
        print('read from csv file:', filename)
        return images,labels

    def __len__(self):
        return  len(self.images)

    def denormalize(self,x_hat):
        mean = [0.485, 0.4565, 0.406]
        std = [0.229, 0.224, 0.225]
        #x_hat = (x-mean)/std
        # x _hat* std +mean
        # mean[3] x_hat[c,h,w] mean [3] ->[3,1,1]
        mean =torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std =torch.tensor(std).unsqueeze(1).unsqueeze(1)
        return (x_hat * std +mean)


    def __getitem__(self, idx):

        img,label =self.images[idx],self.labels[idx]

        x=Image.open(img).convert('RGB')
        #img_array=np.asarray(x)
        #print('img in shape:',img_array.shape)
        tf = transforms.Compose(
            [
                lambda x:Image.open(x).convert('RGB'),  # string path -> image data
                transforms.Resize([int(self.resize*1.25),int(self.resize*1.25)]),
                transforms.RandomRotation(15),
                transforms.CenterCrop(self.resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.4565,0.406],
                                     std=[0.229,0.224,0.225])
            ]
        )
        img = tf(img)
        #print('shape:',img.shape)
        label = torch.tensor(label)
        return img,label

def main():
    import time
    import torchvision

    writer= SummaryWriter(comment = 'test1')

    tf= transforms.Compose([
        transforms.Resize([64,64]),
        transforms.ToTensor()
    ])
    db =torchvision.datasets.ImageFolder(root='pokemon',transform=tf)
    loader = DataLoader(db, batch_size= 32, shuffle = True)
    t = 0
    for x, y in loader:
        writer.add_images('batch_images/batch' + str(t), x, t)
        t += 1
        time.sleep(10)


    # db  = Pokemon('./pokemon',224,'train')
    # x,y = next(iter(db))
    # print('sample:',x.shape,y.shape,y)
    # x=db.denormalize(x)
    # img_array=x
    # print(type(img_array))
    # writer.add_image('image_rotate_1', img_array, 1,dataformats='CHW')
    # writer.close()

    # loader = DataLoader(db,batch_size= 32, shuffle =True)
    # t=0
    # for x, y in loader:
    #     writer.add_images('batch_images/batch'+str(t),db.denormalize(x),t)
    #     t+=1
    #     time.sleep(10)


if __name__ == '__main__':
    main()