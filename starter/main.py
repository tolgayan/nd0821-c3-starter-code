# Put the code for your API here.

from fastapi import FastAPI
from starter.basemodel import Data
from starter.infer import infer

app = FastAPI()


@app.get("/")
def read_root():
    return 'Welcome!'


@app.post("/")
async def create_item(data: Data):
    return infer(data)
