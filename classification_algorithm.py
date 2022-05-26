from glob import glob
import cv2
import os
from tqdm import tqdm
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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.utils import class_weight

class ClassificationModel:
    
    def __init__(self, n_classes=2, n_components=20, input_shape=(224, 224, 3), epochs=100, batch_size=32, learning_rate=0.001, train_choice='nn', splitBy='image', test_size=0.2, stratify=True):
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
        #data generator
        datagen = ImageDataGenerator(rescale=1./255,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     rotation_range=0.3,
                                     zoom_range = [0.7, 1.],
                                     horizontal_flip=True,
                                     vertical_flip=True,
                                     fill_mode='reflect',
                                     shear_range=0.2,
                                     validation_split=0)

        dataset = datagen.flow_from_directory('data',
                                    target_size=(224, 224),
                                    batch_size=32, 
                                    class_mode='binary')

        
        self.classnames_ids = dataset.class_indices
        with open(os.path.join('weights','classes_ids.json'), 'w') as file:
            json.dump(self.classnames_ids, file)

        self.data = []
        self.targets = []
        print('Data Loading + Augmentation...')
        for i, (batch, targ) in enumerate(dataset):
            print(i)
            self.data += list(batch)
            self.targets += list(targ)
            if i > 69:
                break
        print('done!')
        self.data = np.asarray(self.data, dtype=np.float64)
        self.targets = np.asarray(self.targets, dtype=np.float64)
        
        
    def __initNeuralNetworkModel(self):
        self.mobileNetV2 = tf.keras.applications.mobilenet_v2.MobileNetV2(
                            input_shape=self.input_shape,
                            alpha=1.0,
                            include_top=False,
                            weights='imagenet',
                            pooling='avg')
        
        #freez all layers except last 4 layers ;).
        self.mobileNetV2.trainable = False
        for layer in self.mobileNetV2.layers[:-4]:
            layer.trainable = False
        
        #add a classifier to mobilenetv2
        self.model = Sequential()
        self.model.add(self.mobileNetV2)
        #self.model.add(layers.Dropout(0.1))
        self.model.add(layers.Dense(512, activation='relu'))
        self.model.add(layers.Dense(256, activation='relu'))
        #self.model.add(layers.Dropout(0.2))
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

        ##Training:
        recall_1, precision_1=[], []
        recall_0, precision_0=[], []
        accuracy=[]
        skf = StratifiedKFold(n_splits=3, random_state=2022, shuffle=True)
        for i, (train_index, test_index) in enumerate(skf.split(self.data, self.targets)):
            print('--------------------------------------------------------------------------')
            print('------------------------------FOLD---N°{}---------------------------------'.format(i+1))
            X_train, X_test = self.data[train_index], self.data[test_index]
            y_train, y_test = self.targets[train_index], self.targets[test_index]
            
            class_weights = dict(zip(np.unique(y_train), class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)))

            self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, class_weight=class_weights)
            y_pred = np.argmax(self.model(X_test), axis=1)
            print(classification_report(y_test, y_pred))
            
            precision_0 += precision_score(y_test, y_pred, pos_label=0)
            precision_1 += precision_score(y_test, y_pred, pos_label=1)
            recall_0 += recall_score(y_test, y_pred, pos_label=0)
            recall_1 += recall_score(y_test, y_pred, pos_label=1)
            accuracy += accuracy_score(y_test, y_pred)

        print('******Average Stratified KFold Metrics*******')
        print('precision: not flush={:.2f}  flush={:.2f}'.format(np.mean(precision_0), np.mean(precision_1)))
        print('recall:    not flush={:.2f}  flush={:.2f}'.format(np.mean(recall_0), np.mean(recall_1)))
        print('accuracy: {:.2f}'.format(np.mean(precision_0), np.mean(accuracy)))

        self.model.fit(self.data, self.targets,epochs=self.epochs, batch_size=self.batch_size)
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
        
        param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
        grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
        grid.fit(X_train, y_train)
        print("best params : ",grid.best_params_)
        print("best estimator : ",grid.best_estimator_)
        self.y_pred = grid.predict(X_test)
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
    model = ClassificationModel(epochs=4, train_choice='nn', splitBy='image')
    model.train()
    print("--")
    #model.show_classification_report()
    #model.show_results()