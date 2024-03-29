from KMeans import KMeans
from DataReader import DataReader
import matplotlib.pyplot as plt
import numpy as np

data_reader = DataReader('/home/qisens/data/test_2','my')
tr_data, tr_class_labels, tr_subclass_labels, tr_name, tr_classname = data_reader.get_train_data()

input_reader=DataReader('/home/qisens/data/train','my')
data, class_labels, subclass_labels, name, class_name= input_reader.get_train_data()

print(data)
print(class_labels)
print(class_name)

kmeans = KMeans(n_clusters=2,max_iter=1, data=data, classname=class_name)
kmeans.fit(tr_data,tr_class_labels, tr_name)
