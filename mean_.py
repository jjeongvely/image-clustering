# train(input) 에 있는 사진들을 output에 binary file로 저장


import PIL
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import cv2

# data directory
input ="/home/qisens/data/clustering_data/train"
output ="/home/qisens/data/clustering_data/train/test.bin"
imageSize = 32
imageDepth = 3
debugEncodedImage = False
data=[]
labels=[]
name=[]
classname=[]
centroids=[]
dict={}

# show given image on the window for debug
def showImage(r, g, b):
    temp = []
    for i in range(len(r)):
        temp.append(r[i])
        temp.append(g[i])
        temp.append(b[i])
    show = np.array(temp).reshape(imageSize, imageSize, imageDepth)
    plt.imshow(show, interpolation='nearest')
    plt.show()

# convert to binary bitmap given image and write to law output file
def writeBinaray(imagePath, imageFile, label):
    img = Image.open(imagePath)
    #print(img.size)
    img = img.resize((imageSize, imageSize), PIL.Image.ANTIALIAS)
    #print('size ',str(img.size))
    img = (np.array(img))
    #print(img.shape)
    #print(imagePath)
    #print(img.shape)

    if img.shape==(32,32):
        #print(imagePath)

        rgb=img[:,:].flatten()
        #print(rgb.shape)
        if centroids==[]:
            centroids.extend([rgb])
        else:
            centroids[0] = np.mean(np.vstack((centroids[0], [rgb])), axis=0)
        #out = np.array(list(label) + list(rgb), np.uint8)
        """
        temp=[]
        for i in range(len(rgb)):
            temp.append(rgb[i])
        show=np.array(temp).reshape(imageSize, imageSize)
        plt.imshow(show, interpolation='nearest')
        plt.show()
        """
    else:
        #cv2.imshow('test', img)
        #cv2.waitKey(0)
        #r = img[:,:,0].flatten()
        #g = img[:,:,1].flatten()
        #b = img[:,:,2].flatten()
        #d=(r+g+b)/3
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb = img[:, :].flatten()
        if centroids == []:
            centroids.extend([rgb])
        else:
            centroids[0] = np.mean(np.vstack((centroids[0], [rgb])), axis=0)
        #print("r",r.shape)
        #print("g",g.shape)
        #print("b",b.shape)
        #out = np.array(list(label) + list(r) + list(g) + list(b), np.uint8)
    #outputFile.write(out.tobytes())
    # if you want to show the encoded image. set up 'debugEncodedImage' flag
    if debugEncodedImage:
        showImage(r, g, b)

subDirs = os.listdir(input)
numberOfClasses = len(input)

try:
    os.remove(output)
except OSError:
    pass

outputFile = open(output, "wb")
label = -1
totalImageCount = 0
labelMap = []

for subDir in subDirs:
    subDirPath = os.path.join(input, subDir)

    # filter not directory
    if not os.path.isdir(subDirPath):
        continue
    classname.extend([subDir])

    imageFileList = os.listdir(subDirPath)
    label += 1
    labels.extend([label])

    print("writing %3d images, %s" % (len(imageFileList), subDirPath))
    totalImageCount += len(imageFileList)
    labelMap.append([label, subDir])

    centroids=[]
    for imageFile in imageFileList:
        imagePath = os.path.join(subDirPath, imageFile)
        writeBinaray(imagePath, imageFile, label)

    data.extend([centroids[0]])

dict[b'data']=np.array(data, dtype=np.uint8)
dict[b'labels']=labels
dict[b'classname']=classname
dict[b'name']=name

for i in range(len(dict[b'data'])):
    image=dict[b'data'][i]
    image=image.reshape(32,32).astype("uint8")
    image=Image.fromarray(image)
    image.save("/home/qisens/data/clustering_data/train/"+dict[b'classname'][i]+".jpg")


pickle.dump(dict, outputFile, pickle.HIGHEST_PROTOCOL)
outputFile.close()
print("Total image count: ", totalImageCount)
print("Succeed, Generate the Binary file")
print("You can find the binary file : ", output)
print("Label MAP: ", labelMap)
#print(dict)
