import cv2
import numpy as np
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)
from sklearn.utils import shuffle

classes = ['dogs', 'cats']  # The names of the classes.

image_size = 128  # Set the Image Size <- square image 128x128 pixels. Can make not square by modifying the functions below

test_path = '/Users/mk/PycharmProjects/AI/ImageAI/CatDog/testing_data'
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#Path to smaller folder containing testing images
train_path = '/Users/mk/PycharmProjects/AI/ImageAI/CatDog/training_data'
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Path to bigger folder of training images with subfolders containing images named after the classes

def load_train(train_path, image_size, classes):
    images = []
    labels = []
    ids = []
    cls = []


    print('Reading training images')
    for fld in classes:  # assuming data directory has a separate folder for each class, and that each folder is named after the class
        index = classes.index(fld)
        print('Loading {} files (Index: {})'.format(fld, index))
        path = os.path.join(train_path, fld, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Makes it Greyscale to reduce size
            #image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) # Makes the Image B&W when displayed on plt but increases size *3
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            ids.append(flbase)
            cls.append(fld)
    images = np.array(images)
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)

    return images, labels, ids, cls

# images, labels, ids, cls = load_train(train_path, image_size, classes)  #


def load_test(test_path, image_size, classes):
    for class_name in classes:
        path = os.path.join(test_path, class_name, '*g')
        files = sorted(glob.glob(path))

        X_test = []
        X_test_id = []
        print("Reading test images")
        for fl in files:
            flbase = os.path.basename(fl)
            print(fl)
            img = cv2.imread(fl)
            img = cv2.resize(img, (image_size, image_size), cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Makes it Greyscale to reduce size
            X_test.append(img)
            X_test_id.append(flbase)

            ### because we're not creating a DataSet object for the test images, normalization happens here
        X_test = np.array(X_test, dtype=np.uint8)
        X_test = X_test.astype('float32')
        X_test = X_test / 255

    return X_test, X_test_id

# X_test, X_test_id = load_test(test_path, image_size, classes)  #

class DataSet(object):
    def __init__(self, images, labels, ids, cls):
        self._num_examples = images.shape[0]
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0) # divides by the number of colors to get a number between 0-1 for each pixel color

        self._images = images
        self._labels = labels
        self._ids = ids
        self._cls = cls
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def ids(self):
        return self._ids

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):  #Makes batches given a batch size
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1

            # # Shuffle the data (maybe)
            # perm = np.arange(self._num_examples)
            # np.random.shuffle(perm)
            # self._images = self._images[perm]
            # self._labels = self._labels[perm]
            # Start next epoch

            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end], self._ids[start:end], self._cls[start:end]


def read_train_sets(train_path, image_size, classes, validation_size=0):  #Here validation size is a 0-1 indicating how much of the training data you want to use for validation purposes.
    class DataSets(object):
        pass

    data_sets = DataSets()

    images, labels, ids, cls = load_train(train_path, image_size, classes)
    images, labels, ids, cls = shuffle(images, labels, ids, cls)  # shuffle the data

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    validation_ids = ids[:validation_size]
    validation_cls = cls[:validation_size]

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_ids = ids[validation_size:]
    train_cls = cls[validation_size:]

    data_sets.train = DataSet(train_images, train_labels, train_ids, train_cls)
    data_sets.valid = DataSet(validation_images, validation_labels, validation_ids, validation_cls)

    return data_sets

# data_sets = read_train_sets(train_path, image_size, classes)  #data_sets.train and data_sets.valid

def read_test_set(test_path, image_size, classes):
    images, ids = load_test(test_path, image_size, classes)
    return images, ids

# images, ids = read_test_set(test_path, image_size, classes)  #


"""Code for displaying the images to make sure they are fine and checking sizes to make sure they are small enough"""
# fig = plt.figure()
#
# ax1 = plt.subplot2grid((8,6), (0,0), rowspan=4, colspan=3)
# # ax2 = plt.subplot2grid((8,6), (4,0), rowspan=4, colspan=3)
# # ax3 = plt.subplot2grid((8,6), (0,3), rowspan=4, colspan=3)
# # ax4 = plt.subplot2grid((8,6), (4,3), rowspan=4, colspan=3)
# iar = images[55]
# ax1.imshow(iar)
# # ax2.imshow(iar2)
# # ax3.imshow(iar3)
# # ax4.imshow(iar4)
# plt.show()
#
# # print(images[55])
#
# # print(np.size(images[55]))  <-- check the size of image data to make sure it's reasonable
#
# # i = Image.open('ImageAI.CatDog.training_data/cats/*g')
# # iar = np.array(i)
# # print(iar)

