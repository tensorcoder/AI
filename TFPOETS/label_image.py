import os, sys
import tensorflow as tf
import urllib.request


def import_image(address):
    name = "image.jpg"
    fulladdress = address
    urllib.request.urlretrieve(fulladdress, name)

address = str(input('Paste the url of the desired jpg or jpeg image here: '))[1:-1:1]
import_image(address)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# change this as you see fit
image_path = '/Users/mk/PycharmProjects/AI/TFPOETS/image.jpg'

# Read in the image_data
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
               in tf.gfile.GFile("/private/tmp/output_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("/private/tmp/output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    human_string1 = []
    score1 = []
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    predictions = sess.run(softmax_tensor, \
                           {'DecodeJpeg/contents:0': image_data})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]



    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        human_string1.append(human_string)
        score1.append(score)
        print('%s (score = %.5f)' % (human_string, score))

#
# print(score1)
# print(human_string1)
# #print(max(score1), score1.index(max(score1)))



print(human_string1[score1.index(max(score1))][0:-1].upper())

correct = 'k'

while correct[0] != 'y' and correct[0] !='n':
  correct = input('Is this a %s? '%human_string1[score1.index(max(score1))][0:-1])
  if correct[0] == 'y':
      print('i am the master')
      ans = 'k'
      while ans[0] != 'y' and ans[0] != 'n':
          ans = input('Should I add it to the training folder? : ')
          if ans[0] == 'y':
              print('The image has been added to training folder. ')
          elif ans[0] == 'n':
              print('The image has not been added to training folder.')
          else:
              print('What? ')


  elif correct[0] == 'n':
      print('I suck')

  else:
      print('What? ')



