from fastapi import FastAPI
from starlette.responses import JSONResponse
from iris.iris_classifier import IrisClassifier
from iris.models import Iris

app = FastAPI()

@app.get('/train_model', status_code=200)
async def train_model():
    
    return 'Iris classifier is all ready to go!'

@app.get('/healthcheck', status_code=200)
async def healthcheck():
    return 'Iris classifier is all ready to go!'

@app.post('/classify_iris')
def extract_name(iris_features: Iris):
    iris_classifier = IrisClassifier()
    return JSONResponse(iris_classifier.classify_iris(iris_features))

