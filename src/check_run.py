import joblib

import pandas as pd
import numpy as np

best_model = joblib.load('../temp/sentitect-best-model-pkl.pkl')

print(best_model)


data = pd.read_csv('../temp/output-vectorize.csv')
data['vectors'] = data['vectors'].apply(lambda row: [float(t) for t in row.split(';')])

X_train = data.vectors.values
X_train = [np.array(tmp) for tmp in X_train]
X_train = np.array(X_train)
y_train = data.sentiment.values


result = best_model.predict(X_train)
print(np.average(result == y_train))