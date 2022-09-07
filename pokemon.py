import torch
import os,glob
import  random,csv

from torch.utils.data import Dataset

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
        pass

    def __getitem__(self, idx):
        pass

def main():
    db  = Pokemon('./pokemon',224,'train')
if __name__ == '__main__':
    main()