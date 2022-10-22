import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import joblib
import pickle

from iris.v2.models import Iris

class IrisClassifier:
    def __init__(self):
        self.X, self.y = load_iris(return_X_y=True)
        #self.clf = self.train_model()
        self.clf = self.export_model()
        self.iris_type = {
            0: 'setosa',
            1: 'versicolor',
            2: 'virginica'
        }

    def train_model(self) -> LogisticRegression:
        return LogisticRegression(solver='lbfgs',
                                  max_iter=1000,
                                  multi_class='multinomial').fit(self.X, self.y)
    def export_model(self):
        joblib.dump(self.train_model(), "iris_model.pkl")
        print("Model trained and saved!")
        return 0
    
    def load_model(self):
        return joblib.load("iris_model.pkl")

    def classify_iris(self, iris: Iris):
        X = [iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]
        model = self.load_model()
        prediction = model.predict_proba([X])
        print("Classified with loaded model.")
        return {'class': self.iris_type[np.argmax(prediction)],
                'probability': round(max(prediction[0]), 2)}


