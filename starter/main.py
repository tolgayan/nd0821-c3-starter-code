# Put the code for your API here.

import os
import subprocess


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    dvc_output = subprocess.run(
        ["dvc", "pull"], capture_output=True, text=True)
    print(dvc_output.stdout)
    print(dvc_output.stderr)
    if dvc_output.returncode != 0:
        print("dvc pull failed")
    else:
        os.system("rm -r .dvc .apt/usr/lib/dvc")


from fastapi import FastAPI
from starter.starter.basemodel import Data
from starter.starter.infer import infer


app = FastAPI()


@app.get("/")
def read_root():
    return 'Welcome!'


@app.post("/")
async def create_item(data: Data):
    return infer(data)
