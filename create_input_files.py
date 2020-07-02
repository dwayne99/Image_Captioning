# Dependancies
import json
from collections import Counter
from random import seed, choice, sample
import h5py
from tqdm import tqdm
import cv2
from cv2 import imread, resize
import os
import numpy as np


def create_input_files(dataset, json_path, image_folder, captions_per_image,
                       min_word_freq, output_folder, max_len = 100):

    '''
    Creates input files for training, validation, and test data.
    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    '''

    assert dataset in {'coco', 'flickr8k', 'flickr30k'}

    # Read Karpathy JSON
    with open(json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    # iterate over data
    for img in data['images']:
        captions = []

        # iterate over each caption of an image
        for c in img['sentences']:

            # update word frequency
            word_freq.update(c['tokens'])
            
            # don't save caption that exceeds max_len
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        # if all captions of an image don't meet max_len criteria don't save the image
        if len(captions) == 0:
            continue

        # construct the image path
        path = os.path.join(image_folder, img['filename'])

        # Check the value of the split attribute to place image in the desired folder
        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)
            

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create the word map

    # shortlist the words that meet min_word_freq criteria
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v for v, k in enumerate(words,1)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions along with their lengths to JSON files
    seed(123)

    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the num of captions we're sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create a dataset inside HDF5 file to store images
            images = h.create_dataset('Images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions 
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                # Read images
                img = imread(impaths[i])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # if image is grayscale add the depth dimention to it
                if len(img.shape) == 2 :
                    img = img[:,:, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                # Resize the image 
                img = resize(img, (256,256))
                # convert image from H x W x C -->  C x H x W
                img = img.transpose(2,0,1)

                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths, add 2 for 'start' and 'end' tokens
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)


#################################################################################

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(
        dataset='flickr8k',
        json_path = 'data/dataset_flickr8k.json',
        image_folder='data/Flicker8k_Dataset',
        captions_per_image=5,
        min_word_freq=5,
        output_folder='data_output',
        max_len=50
)
