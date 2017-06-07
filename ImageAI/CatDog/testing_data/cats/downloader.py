import urllib.request


# def import_github_img(startindex=1000, endindex=1199, address="https://raw.githubusercontent.com/sankit1/cv-tricks.com/master/Tensorflow-tutorials/tutorial-2-image-\
# classifier/testing_data/cats/"):
#     for index in range(startindex, endindex+1, 1):
#         name = "cat."+str(index)+".jpg"
#         fulladdress = address+name
#         print(name)
#         urllib.request.urlretrieve(fulladdress, name)
#
# # urllib.request.urlretrieve("https://raw.githubusercontent.com/sankit1/cv-tricks.com/master/Tensorflow-tutorials/tutorial-2-image-\
# # classifier/testing_data/cats/cat.1000.jpg", "cat1000.jpg")
# #
#
# import_github_img(1000, 1199, "https://raw.githubusercontent.com/sankit1/cv-tricks.com/master/Tensorflow-tutorials/tutorial-2-image-\
# classifier/testing_data/cats/")
urllib.request.urlretrieve('https://raw.githubusercontent.com/sankit1/cv-tricks.com/master/Tensorflow-tutorials/tutorial-2-image-classifier/testing_data/cats/cat.1013.jpg', 'cat.1013.jpg')