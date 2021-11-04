import uvicorn
from fastapi import FastAPI, File, status
from core.svg import convert_svg_to_lottie, convert_svg_to_lottie_withanimation
from core.svg import convert_svg_to_lottie_def
import shutil
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import UploadFile
import xml.etree.cElementTree as et
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi import FastAPI, HTTPException

from enum import Enum
import cairosvg
import time

app = FastAPI()

origins = [
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "http://localhost",
    "http://localhost:8080"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def is_svg(filename):
    tag = None
    with open(filename, "r") as f:
        try:
            for _, el in et.iterparse(f, ('start',)):
                tag = el.tag
                break
        except et.ParseError:
            pass
    return tag == '{http://www.w3.org/2000/svg}svg'


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return PlainTextResponse(str(exc.detail), status_code=exc.status_code)


@app.post("/uploadsvg/")
def create_upload_file_def(optimize: bool = False, file: UploadFile = File(...)):
    try:
        suffix = Path(file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)
            file.file.close()

            newfile = NamedTemporaryFile(delete=False, suffix=".svg")
            # newfilepath = Path(newfile.name)

        cairosvg.svg2svg(file_obj=open(tmp_path, 'rb'), write_to=newfile.name)

    finally:
        # file.file.close()

        if (is_svg(tmp_path)):
            if not optimize:
                anim = convert_svg_to_lottie_def(str(newfile.name))
            else:
                anim = convert_svg_to_lottie(str(newfile.name))
            # an = json.loads(anim)

            newfilepath = newfile.name
            os.unlink(newfilepath)
            assert not os.path.exists(newfilepath)
            os.unlink(tmp_path)
            assert not os.path.exists(tmp_path)

            return anim
        else:
            error = {"success": False, "message": "Invalid file type"}
            return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=error)


if __name__ == "__main__":
    uvicorn.run("svgtolottie:app", host="0.0.0.0")
