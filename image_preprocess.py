import os
import ProtoPNet.img_aug as aug
from PIL import Image

path_to_imfolder = './data/CUB_200_2011/images/'
path_to_ims = './data/CUB_200_2011/images.txt'
path_to_bound = './data/CUB_200_2011/bounding_boxes.txt'

path_to_train_crop_folder = './data/cub200_cropped/train_cropped/'
path_to_test_crop_folder = './data/cub200_cropped/test_cropped/'

path_to_split = './data/CUB_200_2011/train_test_split.txt'

dict = {}
with open(path_to_ims, 'r') as file:
    for line in file:
        line = line.strip()
        key, loc = line.split(' ')
        dict[int(key)] = {"location": loc}

with open(path_to_split, 'r') as file:
    for line in file:
        line = line.strip()
        key, train = line.split(' ')
        dict[int(key)]['train'] = int(train)

print(dict)



def main():
    ## Crop and save the images

    f = open(path_to_bound, 'r')

    for line in f:
        line = line.strip()  # remove whitespace on ends (newline char)
        im, x, y, width, height = line.split(' ')
        im = int(im)
        x = int(float(x))
        y = int(float(y))
        width = int(float(width))
        height = int(float(height))

        image = Image.open(path_to_imfolder + dict[im]['location'])

        right = x + width
        bottom = y + height

        cropped = image.crop((x, y, right, bottom))
        train = dict[im]['train'] == 0

        path = path_to_train_crop_folder if train else path_to_test_crop_folder
        os.makedirs(path, exist_ok=True)

        save_cropped = path + str(im) + '.jpg'
        print(save_cropped)
        cropped.save(save_cropped)

    ## Augment the images
    aug.main()


if __name__ == "__main__":
    main()