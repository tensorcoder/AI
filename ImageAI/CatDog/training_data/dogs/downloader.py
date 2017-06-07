import urllib.request


def import_github_img(startindex=0, endindex=499, address="https://raw.githubusercontent.com/sankit1/cv-tricks.com/master/Tensorflow-tutorials/tutorial-2-image-\
classifier/training_data/dogs/"):
    for index in range(startindex, endindex+1, 1):
        name = "dog."+str(index)+".jpg"
        fulladdress = address+name
        print(name)
        urllib.request.urlretrieve(fulladdress, name)

# urllib.request.urlretrieve("https://raw.githubusercontent.com/sankit1/cv-tricks.com/master/Tensorflow-tutorials/tutorial-2-image-\
# classifier/testing_data/cats/cat.1000.jpg", "cat1000.jpg")
#

import_github_img(0, 499, "https://raw.githubusercontent.com/sankit1/cv-tricks.com/master/Tensorflow-tutorials/tutorial-2-image-\
classifier/training_data/dogs/")
