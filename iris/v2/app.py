from fastapi import FastAPI
from starlette.responses import JSONResponse
from iris.v2.iris_classifier import IrisClassifier as IrisClassifier
from iris.v2.models import Iris

app = FastAPI()

@app.get('/v2/healthcheck', status_code=200)
async def healthcheck():
    return 'Iris classifier is all ready to go!'

@app.post('/v2/classify_iris')
def extract_name(iris_features: Iris):
    iris_classifier = IrisClassifier()
    return JSONResponse(iris_classifier.classify_iris(iris_features))

