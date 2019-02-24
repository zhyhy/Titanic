#----------------- 导入库函数
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
#----------------- 定义一个函数，处理特征
def PreprocessData(raw_df):
    df = raw_df.drop(['Name'], axis=1)
    age_mean = df['Age'].mean()
    df['Age'] = df['Age'].fillna(age_mean)
    fare_mean = df['Fare'].mean()
    df['Fare'] = df['Fare'].fillna(fare_mean)
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
    x_one_hot_df = pd.get_dummies(data=df, columns=["Embarked"])

    ndarray = x_one_hot_df.values
    Features = ndarray[:, 1:]
    Label = ndarray[:, 0]

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures = minmax_scale.fit_transform(Features)
    #  print(raw_df.isnull().sum())
    return scaledFeatures, Label

#----------------- 加载数据集 train_csv ,test_csv
all_train_df = pd.read_csv('tiantic.csv')
#all_test_df = pd.read_csv('test.csv')
# 提取用到的字段
clos_train = ['Survived','Name','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
#clos_test = ['Name','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
all_train_df = all_train_df[clos_train]
# 将数据以四级的方式分为训练数据和测试数据
msk = np.random.rand(len(all_train_df)) < 0.8
train_df = all_train_df[msk]
test_df = all_train_df[~msk]

print(pd.isnull(train_df['Age']).sum())
train_Features, train_label = PreprocessData(train_df)
test_Features, test_label = PreprocessData(test_df)

#----------------- 搭建神经网络
module = Sequential()

module.add(Dense(units=16,input_dim=9,
                 activation='relu'))
module.add(Dense(units=32,input_dim=16,
                 activation='relu'))
module.add(Dense(units=64,input_dim=32,
                 activation='relu'))
module.add(Dense(units=16,input_dim=64,
                 activation='relu'))
module.add(Dense(units=1,input_dim=64,
                 activation='sigmoid'))

print(module.summary())

# -------------------
#         训练       -
#--------------------
module.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
module.fit(x = train_Features,
           y = train_label,
           validation_split=0.1,
           epochs=50,batch_size=100,verbose=2)

score = module.evaluate(test_Features,test_label)
print(score[1])