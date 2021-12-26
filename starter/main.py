# Put the code for your API here.

from fastapi import FastAPI
from starter.basemodel import Data
from starter.infer import infer
import os


app = FastAPI()


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


@app.get("/")
def read_root():
    return 'Welcome!'


@app.post("/")
async def create_item(data: Data):
    return infer(data)
