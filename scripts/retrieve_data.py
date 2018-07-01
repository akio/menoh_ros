#!/usr/bin/env python
import os
import sys
import urllib
import rospkg

rospack = rospkg.RosPack()
pkg_dir = rospack.get_path('menoh_ros')
data_dir = os.path.join(pkg_dir, 'data')

def download(address, target):
    print('downloading' + address + ' to ' + target)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    urllib.urlretrieve(address, os.path.join(data_dir, target))

download('https://www.dropbox.com/s/bjfn9kehukpbmcm/VGG16.onnx?dl=1',
         'VGG16.onnx')
download('https://raw.githubusercontent.com/HoldenCaulfieldRye/caffe/master/data/ilsvrc12/synset_words.txt',
         'synset_words.txt')
download('https://upload.wikimedia.org/wikipedia/commons/5/54/Light_sussex_hen.jpg',
         'Light_sussex_hen.jpg')

