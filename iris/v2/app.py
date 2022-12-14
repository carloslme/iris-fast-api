from fastapi import FastAPI
from starlette.responses import JSONResponse
from iris.v2.iris_classifier import IrisClassifier as IrisClassifier
from iris.v2.models import Iris

app = FastAPI()
iris_classifier = IrisClassifier()

@app.get('/v2/healthcheck', status_code=200)
async def healthcheck():
    return 'Iris classifier is all ready to go!'

@app.post('/v2/classify_iris')
def extract_name(iris_features: Iris):
    print(type(iris_classifier))
    return JSONResponse(iris_classifier.classify_iris(iris_features))


@app.on_event("startup")
async def startup():
    iris_classifier.load_model()
