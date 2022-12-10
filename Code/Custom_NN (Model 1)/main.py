import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from train import *

# configs path
configYamlPath = "configs/" 
# Replace this with a yaml file 
configFile='128-128-relu.yaml'

if __name__ == "__main__":

    

    # Load data
    data = pd.read_csv('features_3_sec.csv')
    data.head()
    data = data.drop(labels='filename',axis=1)
    class_list = data.iloc[:, -1]
    convertor = LabelEncoder()
    # split train-test data
    y = convertor.fit_transform(class_list)
    fit = StandardScaler()
    X = fit.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    y_train, y_test = one_hot_encoding(y_train), one_hot_encoding(y_test)
    # Load the configuration from the corresponding yaml file. Specify the file path and name
    config = util.load_config(configYamlPath + configFile) # Set configYamlPath, configFile  in constants.py

    # Create a Neural Network object which will be our model
    model = Neuralnetwork(config)

    # train the model. Use train.py's train method for this
    model = train(model, x_train, y_train, x_test, y_test, config, configFile)
    tqdm.write('Completed training.')