from glob import glob
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
import pickle
import json
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV,cross_validate,cross_val_score,KFold

import shutil

import time


class ClassificationModel:
    
    def __init__(self, n_classes=2, n_components=20, input_shape=(224, 224, 3), epochs=10, batch_size=32, learning_rate=0.001, train_choice='composed', splitBy='random', test_size=0.2, stratify=True):
        self.n_components = None
        self.mobileNetV2 = None
        self.model = None
        self.svm = None
        self.X_train, self.X_test, self.y_train, self.y_test = [], [], [], []
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.classnames_ids = {}
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.train_choice = train_choice #composed model / neuralNetwork
        self.splitBy = splitBy #random / image
        self.stratify = stratify
        self.test_size = test_size

    def __features_transformation(self, features):
        #pca = PCA(n_components= self.n_components)
        extracted_features = self.mobileNetV2.predict(features)
        #reduced_features = pca.fit_transform(extracted_features)
        return extracted_features 

    def __loadDataSet(self):
        
        '''    
        resize_shape = self.input_shape[:2]

        for i, class_name in enumerate(glob(os.path.join('dataset','*'))):
            #self.classnames_ids[i] = class_name.split('\\')[-1]  #register new id -> class name
            if i == 1:
                class_id = 1
                self.classnames_ids[1] = 'FLUSH'
            else:
                class_id = 0
                self.classnames_ids[0] = 'OTHER'  #register new id -> class name
            
            #load training data
            train_path = os.path.join(class_name, 'train')
            for frame_path in glob(os.path.join(train_path,'*.bmp')):
                frame = cv2.imread(frame_path)
                frame = cv2.resize(frame, resize_shape, interpolation = cv2.INTER_AREA) #resize the frame
                frame = frame/255
                self.X_train.append(frame)
                self.y_train.append(class_id)

            #load test data
            test_path = os.path.join(class_name, 'test')
            for frame_path in glob(os.path.join(test_path, '*.bmp')):
                frame = cv2.imread(frame_path)
                frame = cv2.resize(frame, resize_shape, interpolation = cv2.INTER_AREA) #resize the frame
                frame = frame/255
                self.X_test.append(frame)
                self.y_test.append(class_id)
        
        #split frames randomly and not by cup images (split already done during the dataset generation)
        if self.splitBy == 'random':
            dataset = self.X_train + self.X_test
            targets = self.y_train + self.y_test
            if self.stratify:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(dataset, targets, test_size=self.test_size, stratify=targets, random_state=2022)
            else:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(dataset, targets, test_size=self.test_size, random_state=2022)

        #convert train/test data into numpy arrays
        self.X_train = np.asarray(self.X_train)
        self.X_test = np.asarray(self.X_test)
        

        with open(os.path.join('weights','classes_ids.json'), 'w') as file:
            json.dump(self.classnames_ids, file)
        '''
        # creating train / val /test
        root_dir = 'data/'
        new_root = 'AllDataset/'
        classes = ['hgs_flush', 'hgs_not_flush']
        resize_shape = self.input_shape[:2]

        for cls in classes:
            os.makedirs(root_dir + new_root+ cls + '/train/'  ,exist_ok=True)
            os.makedirs(root_dir +new_root +  cls + '/test/' ,exist_ok=True)
        for cls in classes:
            src = root_dir + cls # folder to copy images from
            allFileNames = os.listdir(src)
            np.random.shuffle(allFileNames)

            ## here 0.75 = training ratio , (0.95-0.75) = validation ratio , (1-0.95) = training ratio  
            train_FileNames,test_FileNames = np.split(np.array(allFileNames),[int(len(allFileNames)*0.95)])
            train_FileNames = [src+'/'+ name for name in train_FileNames]
            test_FileNames = [src+'/' + name for name in test_FileNames]
            ## Copy pasting images to target directory
            for name in train_FileNames:
                shutil.copy(name, root_dir + new_root+cls+'/train/' )
            for name in test_FileNames:
                shutil.copy(name,root_dir + new_root+cls+'/test/' )
        
        

        for i, class_name in enumerate(glob(os.path.join('data/AllDataset','*'))):
            #self.classnames_ids[i] = class_name.split('\\')[-1]  #register new id -> class name
            if i == 1:
                class_id = 1
                self.classnames_ids[1] = 'not_flush'
            else:
                class_id = 0
                self.classnames_ids[0] = 'flush'  #register new id -> class name

            train_path = os.path.join(class_name, 'train')
            for frame_path in glob(os.path.join(train_path,'*.bmp')):
                    frame = cv2.imread(frame_path)
                   
                    if frame is None:
                        print('Wrong path:', frame_path)
                    else :
                        frame = cv2.resize(frame, resize_shape, interpolation = cv2.INTER_AREA) #resize the frame
                        frame = frame/255
                        self.X_train.append(frame)
                        self.y_train.append(class_id)

            test_path = os.path.join(class_name,'test')
            for frame_path in glob(os.path.join(test_path, '*.bmp')):
                    frame = cv2.imread(frame_path)
                    if frame is None:
                        print('Wrong path:', frame_path)
                    else :
                        frame = cv2.resize(frame, resize_shape, interpolation = cv2.INTER_AREA) #resize the frame
                        frame = frame/255
                        self.X_test.append(frame)
                        self.y_test.append(class_id)

        if self.splitBy == 'random':
            dataset = self.X_train + self.X_test
            targets = self.y_train + self.y_test
            if self.stratify:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(dataset, targets, test_size=self.test_size, stratify=targets, random_state=2022)
            else:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(dataset, targets, test_size=self.test_size, random_state=2022)

        #convert train/test data into numpy arrays
        self.X_train = np.asarray(self.X_train)
        self.X_test = np.asarray(self.X_test)

        with open(os.path.join('weights','classes_ids.json'), 'w') as file:
            json.dump(self.classnames_ids, file)




        
    def __initNeuralNetworkModel(self):
        self.mobileNetV2 = tf.keras.applications.mobilenet_v2.MobileNetV2(
                            input_shape=self.input_shape,
                            alpha=1.0,
                            include_top=False,
                            weights='imagenet',
                            pooling='avg')
        
        #freez all layers except last 4 layers ;).
        self.mobileNetV2.trainable = True
        for layer in self.mobileNetV2.layers[:-4]:
            layer.trainable = False
        
        #add a classifier to mobilenetv2
        self.model = Sequential()
        self.model.add(self.mobileNetV2)
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dropout(0.1))
        self.model.add(layers.Dense(self.n_classes, activation='softmax'))
        #model.summary()

        #compile the model
        optimizer = Adam(learning_rate=self.learning_rate)
        loss = SparseCategoricalCrossentropy()
        metrics = SparseCategoricalAccuracy()
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def __initComposedModel(self):
        self.mobileNetV2 = tf.keras.applications.mobilenet_v2.MobileNetV2(
                            input_shape=self.input_shape,
                            alpha=1.0,
                            include_top=False,
                            weights='imagenet',
                            pooling='avg')

    
        
        self.svm = SVC(C=1.0, kernel='poly', degree=4, gamma='scale', coef0=0.9, shrinking=True, probability=False, tol=0.001, cache_size=200,
                       class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=2022)
        

    def __execute_neuralNetworkPipeline(self):
        self.__initNeuralNetworkModel()
        self.__loadDataSet()
        y_train = np.asarray(self.y_train, dtype=np.float64)
        self.model.fit(epochs=self.epochs, batch_size=self.batch_size , x=self.X_train, y=y_train)
        self.y_pred = np.argmax(self.model(self.X_test), axis=1)
        #save the model
        os.makedirs(os.path.join('weights', 'nn'), exist_ok=True)
        self.model.save_weights(os.path.join('weights', 'nn', 'nn_model'))

    
    


    def __execute_composedModelPipeline(self):
        self.__initComposedModel()
        self.__loadDataSet()
        y_train = np.asarray(self.y_train, dtype=np.float64)
        X_train = self.__features_transformation(self.X_train)
        X_test = self.__features_transformation(self.X_test)
        #grid search
        """
        param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
        grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
        grid.fit(X_train, y_train)
        print("best params : ",grid.best_params_)
        print("best estimator : ",grid.best_estimator_)
        self.y_pred = grid.predict(X_test)
        #save model
        os.makedirs(os.path.join('weights', 'composed'), exist_ok=True)
        pickle.dump(grid.best_estimator_, open(os.path.join('weights', 'composed', 'composed_model.sav'), 'wb'))

        """
        #Random search
        params = {"C": np.arange(2, 10, 2),
             "gamma": np.arange(0.1, 1, 0.2)}
        grid = RandomizedSearchCV(SVC(), params, refit = True, random_state=0)
        grid.fit(X_train, y_train)
        print("best params : ",grid.best_params_)
        print("best estimator : ",grid.best_estimator_)
        self.y_pred = grid.predict(X_test)
        #print(self.y_pred)


        
        kf=KFold(n_splits=5)
        #scoring = "neg_root_mean_squared_error"
        #polyscores = cross_validate(grid, X_train, y_train, scoring=scoring, return_estimator=True)
        #print(polyscores["test_score"].mean())
        score=cross_val_score(grid,X_train,y_train,cv=kf)
        #print("Cross Validation Scores are {}".format(score))
        #print("Average Cross Validation score :{}".format(score.mean()))

        #save model
        os.makedirs(os.path.join('weights', 'composed'), exist_ok=True)
        pickle.dump(grid.best_estimator_, open(os.path.join('weights', 'composed', 'composed_model.sav'), 'wb'))
        
        
    def show_classification_report(self):
        print(classification_report(self.y_test, self.y_pred))
            

    def show_results(self):
        for i, frame in enumerate(self.X_test):
            class_id_pred = self.classnames_ids[self.y_pred[i]]
            class_id_true = self.classnames_ids[int(self.y_test[i])]
            plt.title('True: {} Predicted: {}'.format(class_id_true, class_id_pred))
            plt.imshow(frame)
            plt.show()
    
    

    def train(self):
        if self.train_choice == 'nn':
            self.__execute_neuralNetworkPipeline()
        elif self.train_choice == 'composed':
            self.__execute_composedModelPipeline()
          
    
if __name__ == '__main__':
    model = ClassificationModel(epochs=10, train_choice='nn', splitBy='random')
    start = time.time()
    model.train()
    model.show_classification_report()
    end = time.time()
    inf_time = end-start
    print('inference time execution :',inf_time)
    #model.show_results()