# iris-fast-api
This is an example to explain how to serve a ML model as an API by using FastAPI

# Serving Iris Classifier with FastAPI

## Data

The Iris dataset is a simple, yet popular dataset consisting of 150 observations. Each observation captures the sepal length, sepal width, petal length, petal width of an iris (all in cm) and the corresponding iris subclass (one of *setosa, versicolor, virginica*).

![](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Machine+Learning+R/iris-machinelearning.png)

# Setup
* Create a virtual environment with:
```
python3 -m venv venv
```

* Activate the virtual environment

```
source venv/bin/activate
```

# Install the other libraries
Run the following command to install the other libraries.

```
pip install -r requirements.txt
```

# Usage
## Run FastAPI
Run next command to start the api locally

```
uvicorn iris.app:app --port 5000 --reload
```
## Test request
The input is a JSON with the following fields:

* sepal_l
* sepal_w
* petal_l
* petal_w

Corresponding values are the measurements in cm.

Example request:

```
curl 'http://localhost:8080/iris/classify_iris' -X POST -H 'Content-Type: application/json' -d '{"sepal_l": 5, "sepal_w": 2, "petal_l": 3, "petal_w": 4}'
```