# train과 test의 binary file을 불러오고, fit 하기

from KMeans_ import KMeans
from DataReader_ import DataReader

# test할 데이터들의 binary file을 읽어옴
data_reader = DataReader('/home/qisens/data/clustering_data/test','my')
tr_data, tr_class_labels, tr_subclass_labels, tr_name, tr_classname = data_reader.get_train_data()

# 기준점이 될 사진들의 centroid값을 읽어옴
input_reader=DataReader('/home/qisens/data/clustering_data/train','my')
data, class_labels, subclass_labels, name, class_name= input_reader.get_train_data()


kmeans = KMeans(n_clusters=6,max_iter=1, data=data, classname=class_name)
kmeans.fit(tr_data,tr_class_labels, tr_name)
