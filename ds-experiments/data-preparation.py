#let us assume that the dataset exists

import os
import pandas as pd

current_dir = os.path.dirname(__file__)
print(current_dir)



def retrieve_train_data(input_dir):
    data = []
    for sub_dir in enumerate(['neg', 'pos']):

        complete_path = os.path.join(input_dir,sub_dir[1])
        for file in os.listdir(complete_path):
            #print(os.path.join(complete_path, file))

            with open(os.path.join(complete_path, file),'r') as f:
                content = f.read(-1)
                data.append([content, sub_dir[0]])

    return data

def retrieve_predict_data(input_dir):
    data = []
    for sub_dir in enumerate(['unsup']):

        complete_path = os.path.join(input_dir,sub_dir[1])
        for file in os.listdir(complete_path):
            print(os.path.join(complete_path, file))

            with open(os.path.join(complete_path, file),'r') as f:
                content = f.read(-1)
                data.append(content)

    return data

print('Training data preparation')
train_dir = os.path.join(current_dir, 'dataset/aclImdb/train')
train_data = retrieve_train_data(train_dir)
df = pd.DataFrame(train_data,columns=['review','sentiment'])
df.to_csv('train-data.csv',index=False)

print('Test data preparation')
test_dir = os.path.join(current_dir, 'dataset/aclImdb/test')
test_data = retrieve_train_data(test_dir)
df = pd.DataFrame(test_data,columns=['review','sentiment'])
df.to_csv('test-data.csv',index=False)

print('Predict data preparation')
predict_dir = os.path.join(current_dir, 'dataset/aclImdb/train')
predict_data = retrieve_predict_data(predict_dir)
df = pd.DataFrame(predict_data,columns=['review'])
df.to_csv('predict-data.csv',index=False)

