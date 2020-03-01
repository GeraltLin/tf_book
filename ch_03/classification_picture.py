# -*- coding: utf-8 -*-

# @Time : 2020/3/1 20:02
# @Author : linwenxing
# @File : classification_picture
# @Software: PyCharm


import sys

nets_path = r'slim'
if nets_path not in sys.path:
    sys.path.insert(0, nets_path)
else:
    print('already add slim')

import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
from nets.nasnet import pnasnet
import numpy as np
from datasets import imagenet

slim = tf.contrib.slim

tf.reset_default_graph()

image_size = pnasnet.build_pnasnet_mobile.default_image_size
labels = imagenet.create_readable_names_for_imagenet_labels()
print(len(labels), labels)


def getone(onestr):
    return onestr.replace(',', ' ')


with open('中文标签.csv', 'r+') as f:
    labels = list(map(getone, list(f)))
    print(len(labels), type(labels), labels[:5])


sample_images = ['hy.jpg','ps.jpg','72.jpg']
input_imgs = tf.placeholder(tf.float32,[None,image_size,image_size,3])

x1 = 2*(input_imgs/255.0)-1.0

arg_scope = pnasnet.pnasnet_mobile_arg_scope()  #模型命名空间

with slim.arg_scope(arg_scope):
    logits, end_points =pnasnet.build_pnasnet_mobile(x1,num_classes=1001,is_training=False)
    prob = end_points['Predictions']
    y = tf.argmax(prob,axis = 1)                          #获得结果的输出节点

checkpoint_file = r'pnasnet-5_mobile_2017_12_13/model.ckpt'
saver = tf.train.Saver()

with tf.Session() as sess:                              #建立会话
    saver.restore(sess, checkpoint_file)                #载入模型

    def preimg(img):                                    #定义图片预处理函数
        ch = 3
        if img.mode=='RGBA':                            #兼容RGBA图片
            ch = 4

        imgnp = np.asarray(img.resize((image_size,image_size)),
                          dtype=np.float32).reshape(image_size,image_size,ch)
        return imgnp[:,:,:3]
    #获得原始图片与预处理图片
    batchImg = [ preimg( Image.open(imgfilename) ) for imgfilename in sample_images ]
    orgImg = [  Image.open(imgfilename)  for imgfilename in sample_images ]

    yv,img_norm = sess.run([y,x1], feed_dict={input_imgs: batchImg})    #输入到模型

    print(yv,np.shape(yv))                                  #显示输出结果
    def showresult(yy,img_norm,img_org):                    #定义显示图片函数
        plt.figure()
        p1 = plt.subplot(121)
        p2 = plt.subplot(122)
        p1.imshow(img_org)# 显示图片
        p1.axis('off')
        p1.set_title("organization image")

        p2.imshow((img_norm * 255).astype(np.uint8))# 显示图片
        p2.axis('off')
        p2.set_title("input image")

        plt.show()

        print(yy,labels[yy])

    for yy,img1,img2 in zip(yv,batchImg,orgImg):            #显示每条结果及图片
        showresult(yy,img1,img2)
