# train의 centroid들과 test data들을 비교하여 분류

import os

import numpy as np
import copy
import cv2

from tqdm import tqdm
class KMeans:
    
    def __init__(self,n_clusters=10,max_iter=500, data=[], classname=[]):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.data=data
        self.classname=classname
        self.loss_per_iteration = []

    def init_centroids(self):
        #np.random.seed(np.random.randint(0,100000))
        self.centroids = []
        for i in range(self.n_clusters):
            #rand_index = np.random.choice(range(len(self.fit_data)))
            #self.centroids.append(self.fit_data[rand_index])
            #data_reader.plot_img(self.fit_data[rand_index])
            self.centroids.append(self.data[i])

    def init_clusters(self):
        self.clusters = {'data':{i:[] for i in range(self.n_clusters)}}
        self.clusters['labels']={i:[] for i in range(self.n_clusters)}
        self.clusters['name']={i:[] for i in range(self.n_clusters)}

    def fit(self,fit_data,fit_labels, fit_name):
        self.fit_data = fit_data
        self.fit_labels = fit_labels
        self.predicted_labels = [None for _ in range(self.fit_data.shape[0])]
        self.init_centroids() # n_clusters만큼 data에서 뽑아 centroids생성
        self.iterations = 0
        old_centroids = [np.zeros(shape=(fit_data.shape[1],)) for _ in range(self.n_clusters)] # n_clusters 만큼 zeros
        while not self.converged(self.iterations,old_centroids,self.centroids): # iter가 max_iter보다 크거나, old와 cen의 벡터길이<=le-20인경우 True
            old_centroids = copy.deepcopy(self.centroids) # 객체를 복사(즉, 두개 객체)
            self.init_clusters()
            for j,sample in tqdm(enumerate(self.fit_data)):
                #print(fit_name[j])
                #image = sample.reshape(3, 32, 32).transpose(1, 2, 0).astype("uint8")
                #cv2.imshow('test', image)
                #cv2.waitKey(0)
                min_dist = float('inf') # 양의 무한대
                for i,centroid in enumerate(self.centroids):
                    dist = np.linalg.norm(sample-centroid) # sample은 전체데이터, centroid는 랜덤으로 뽑은 데이터
                    if dist<min_dist:
                        min_dist = dist
                        self.predicted_labels[j] = i # 차이가 가장 적을때의 i값(랜덤으로 뽑은 데이터 인덱스)을 예측값으로
                if self.predicted_labels[j] is not None: # 랜덤으로 뽑은 각 index에 data와 label 저장
                    self.clusters['data'][self.predicted_labels[j]].append(sample)
                    self.clusters['labels'][self.predicted_labels[j]].append(self.predicted_labels[j])
                    self.clusters['name'][self.predicted_labels[j]].append(fit_name[j])
            self.reshape_cluster()
            #self.update_centroids()
            self.calculate_loss()
            print("\nIteration:",self.iterations,'Loss:',self.loss,'Difference:',self.centroids_dist)
            self.iterations+=1


        #print(self.clusters)
        outputPath = "/home/qisens/data/result/"
        testPath="/home/qisens/data/clustering_data/test/test/"
        imageFileList = os.listdir(testPath)
        #subDirs = os.listdir(testPath)
        for i in range(self.n_clusters):
            output=outputPath+self.classname[i]
            # print(self.classname[i])
            # print(output)
            if not os.path.isdir(output):
                os.mkdir(output)
            for j in range(len(self.clusters['name'][i])):
                savePath=output+'/'+self.clusters['name'][i][j]
                # print(savePath)
                for imageFile in imageFileList:
                    if imageFile == self.clusters['name'][i][j]:
                        imgPath=testPath+self.clusters['name'][i][j]
                        #print(savePath)
                        # print(imgPath)
                        img=cv2.imread(imgPath,1)
                        #cv2.imshow('test', img)
                        #cv2.waitKey(0)
                        cv2.imwrite(savePath,img)
        """
        outputFile = open(outputPath+"result.bin", "wb")
        #self.result=self.centroids
        #self.result={}
        #self.result['data']=self.centroids
        #self.result['classname']=self.classname

        for i in range(len(self.result['data'])):
            image = self.result['data'][i]
            image = image.reshape(32, 32).astype("uint8")
            image = Image.fromarray(image)
            image.save("/home/qisens/data/result/" + str(i) + ".jpg")

        pickle.dump(self.result, outputFile, pickle.HIGHEST_PROTOCOL)
        outputFile.close()
        """
        self.calculate_accuracy()

    def update_centroids(self): # cluster가 비어있으면 random으로 채워주고, 비어있지 않으면 랜덤으로 뽑은것과 비슷하다고 예측한 결과들을 centroids에 저장
        for i in range(self.n_clusters):
            cluster = self.clusters['data'][i]
            if not cluster.any():
                self.centroids[i] = self.fit_data[np.random.choice(range(len(self.fit_data)))]
            else:
                self.centroids[i] = np.mean(np.vstack((self.centroids[i],cluster)),axis=0) #centroids와 cluster를 수직으로 붙여서 평균
    
    def reshape_cluster(self):
        for id,mat in list(self.clusters['data'].items()):
            self.clusters['data'][id] = np.array(mat) # list를 array로

    def converged(self,iterations,centroids,updated_centroids):
        self.centroids_dist = np.linalg.norm(np.array(updated_centroids)-np.array(centroids)) # 벡터길이 계산
        if self.centroids_dist<=1e-20:
            print("Converged! With distance:",self.centroids_dist)
            return True
        return False

    def calculate_loss(self): # 업데이트한 centroids(예측한결과와 랜덤값의 평균)과 예측한 결과들을 저장한 clusters의 차이
        self.loss = 0
        for key,value in list(self.clusters['data'].items()):
            if value is not None:
                for v in value:
                    self.loss += np.linalg.norm(v-self.centroids[key])
        self.loss_per_iteration.append(self.loss)
    
    def calculate_accuracy(self):
        self.clusters_labels = []
        self.clusters_info = []
        self.clusters_accuracy = []
        for clust,labels in list(self.clusters['labels'].items()): # 랜덤으로 값뽑은 값과 비슷하다고 예측된 이미지들의 label
            # print(clust)
            # print(labels)
            if labels==[]:
                self.clusters_labels.append(clust)
                continue
            if isinstance(labels[0],(np.ndarray)):
                labels = [l[0] for l in labels]
            occur = 0
            max_label = max(set(labels), key=labels.count) # 각 cluster당 label의 수가 가장많은것을 max_label로
            self.clusters_labels.append(max_label)
            for label in labels:
                if label == max_label:
                    occur+=1
            acc = occur/len(list(labels)) # max_label의 수 / 랜덤data와 비슷하다고 판단된 총 label 수
            self.clusters_info.append([max_label,occur,len(list(labels)),acc])
            self.clusters_accuracy.append(acc)
            self.accuracy = sum(self.clusters_accuracy)/self.n_clusters
        self.labels_ = []
        # print(self.clusters_labels)
        for i in range(len(self.predicted_labels)):
            self.labels_.append(self.clusters_labels[self.predicted_labels[i]])
        print('[cluster_label,no_occurence_of_label,total_samples_in_cluster,cluster_accuracy]',self.clusters_info)
        print('Accuracy:',self.accuracy)