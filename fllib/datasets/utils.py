from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, images, labels, transform_x=None, transform_y=None):
        self.images = images
        self.labels = labels
        self.transform_x = transform_x
        self.transform_y = transform_y
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data, label = self.images[index], self.labels[index]
        if self.transform_x is not None:
            data = self.transform_x(Image.open(data))
        else:
            data = Image.open(data)
        if self.transform_y is not None:
            label = self.transform_y(label)
        return data, label


class TransformDataset(Dataset):
    def __init__(self, images, labels, caption_code, caption_length, same_id_caption_code, same_id_caption_length, transform_x=None, transform_y=None):
        self.samples = images
        self.labels = labels
        self.caption_code = caption_code
        self.caption_length = caption_length
        self.same_id_caption_code = same_id_caption_code
        self.same_id_caption_length = same_id_caption_length
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.samples[idx]
        label = self.labels[idx]
        caption_code = self.caption_code[idx]
        caption_length = self.caption_length[idx]
        same_id_caption_code = self.same_id_caption_code[idx]
        same_id_caption_length = self.same_id_caption_length[idx]
        
        if self.transform_x:
            data = self.transform_x(data)
        if self.transform_y:
            label = self.transform_y(label)

        return data, label, caption_code, caption_length, same_id_caption_code, same_id_caption_length