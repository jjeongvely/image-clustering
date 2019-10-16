# test(input)에 있는 사진들을 output에 binary file로 저장


import PIL
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import cv2

# data directory
input ="/home/qisens/data/clustering_data/test"
output ="/home/qisens/data/clustering_data/test/test.bin"
imageSize = 32
imageDepth = 3
data=[]
labels=[]
name=[]
classname=[]
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
    img = img.resize((imageSize, imageSize), PIL.Image.ANTIALIAS)
    img = (np.array(img))

    if img.shape==(32,32):
        rgb=img[:,:].flatten()
        label=[label]
        data.extend([rgb])
        labels.extend(label)
        name.extend([imageFile])
    else:
        label = [label]
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb = img[:, :].flatten()

        data.extend([rgb])
        labels.extend(label)
        name.extend([imageFile])


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

    print("writing %3d images, %s" % (len(imageFileList), subDirPath))
    totalImageCount += len(imageFileList)
    labelMap.append([label, subDir])

    for imageFile in imageFileList:
        imagePath = os.path.join(subDirPath, imageFile)
        writeBinaray(imagePath, imageFile, label)


dict[b'data']=np.array(data, dtype=np.uint8)
dict[b'labels']=labels
dict[b'name']=name
dict[b'classname']=classname
pickle.dump(dict, outputFile, pickle.HIGHEST_PROTOCOL)
outputFile.close()

print("Total image count: ", totalImageCount)
print("Succeed, Generate the Binary file")
print("You can find the binary file : ", output)
print("Label MAP: ", labelMap)
