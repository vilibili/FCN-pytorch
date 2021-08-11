import os
import tqdm
import imageio
import numpy as np
from torch.utils.data import Dataset

class datareader(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, image_list_path):
        'Initialization'
        self.image_path = r'dataset\images'
        self.mask_path = r'dataset\masks'
        self.image_list = self.get_image_list(image_list_path)
        self.images, self.labels = self.get_Images()

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.images)

  def __getitem__(self, index):
        'Generates one sample of data'

        x = np.moveaxis(self.images[index], -1, 0)
        y = self.labels[index]

        return x, y

  def get_image_list(self, path):
      f = open(path, 'r')
      text = f.readlines()
      f.close()
      imagelist = []
      for name in text:
          name = name.replace('\n', '')
          imagelist.append(name)
      return imagelist

  def get_Images(self):
      images = []
      masks = []
      for index in tqdm.trange(len(self.image_list)):
          image, mask = self.read_data(index)
          images += image
          masks += mask

      images = np.array(images, dtype=np.uint8)
      masks = np.array(masks, dtype=np.uint8)
      return images, masks

  def read_data(self, index):
      images = []
      masks = []

      name = self.image_list[index].split(" ")[0]
      image = imageio.imread(os.path.join(self.image_path, name + '.jpeg'), as_gray=False, pilmode="RGB")
      mask = imageio.imread(os.path.join(self.mask_path, name + '.png'))

      images.append(image)
      mask[mask == 255] = 1
      masks.append(mask)
      return images, masks
