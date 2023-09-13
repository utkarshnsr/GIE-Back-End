from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi.concurrency import run_in_threadpool
from classifier.inference import run_inference, make_model
from keras.utils import load_img
from io import BytesIO, StringIO
import requests
from PIL import Image
from pydantic import BaseModel
import random
import os
from starlette.responses import Response
import pandas as pd
from data_extractor import DataExtractor
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
import easyocr
from line_plots.line_plot import LinePlot

app = FastAPI()


# Allow CORS to prevent errors when running on different domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLRequest(BaseModel):
    url: str

classes = ["BarGraph", "LineGraph", "PieChart", "ScatterGraph"]
model_path = "classifier\model1.keras"  # can also use model2.keras
image_size = (224, 224)
model = make_model(input_shape=image_size + (3,), num_classes=len(classes))
model.load_weights(model_path)
reader = easyocr.Reader(['en'])
lp = LinePlot()
de = DataExtractor(reader)


@app.get("/")
async def hello():
    return {"message": "working, correctly!"}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        os.makedirs('inputs/', exist_ok=True)
        contents = await file.read()
        img_to_save = load_img(BytesIO(contents))
        img = load_img(BytesIO(contents), target_size=(224, 224))
        img_to_save.save('inputs/'+file.filename)
        classifier_result = run_inference(model, img)
        if "scatter" in str(list(classifier_result.items())[0][0]).lower():
            de.run('inputs/'+file.filename, scatter=True, line=False)
        elif "line" in str(list(classifier_result.items())[0][0]).lower():
            img_path = lp.run_dotted('inputs/'+file.filename)
            de.run(img_path, scatter=False, line=True)
        else:
            return {"message":"Only ScatterPlots and LineGraphs are supported right now", "classifier_result": classifier_result}
    except Exception as e:
        print('Exception in uploadFile:',e)
        return {"message": "There was an error uploading the file"}

    return {"message": f"successfully uploaded {file.filename}", "classifier_result": classifier_result} 

@app.post("/download")
async def download_image(request: URLRequest):
    try:
        headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36'}
        response = requests.get(request.url, headers=headers, stream=True)
        image_to_save = Image.open(BytesIO(response.content)).convert('RGB')
        image = image_to_save.resize((224, 224))
        classifier_result = run_inference(model, image)
        img_no = str(random.randint(1,1000000))
        if os.path.exists('inputs/downloaded_image_'+img_no+'.jpg'):
            img_no = str(random.randint(1,1000000))
        image_to_save.save("inputs/downloaded_image_"+img_no+'.jpg')
        if "scatter" in str(list(classifier_result.items())[0][0]).lower():
            de.run("inputs/downloaded_image_"+img_no+'.jpg', scatter=True, line=False)
        elif "line" in str(list(classifier_result.items())[0][0]).lower():
            img_path = lp.run_dotted("inputs/downloaded_image_"+img_no+'.jpg')
            de.run(img_path, scatter=False, line=True)
        else:
            return {"message":"Only ScatterPlots and LineGraphs are supported right now", "classifier_result": classifier_result}
    except Exception as e:
        print('Exception in downloading image:',e)
        return {"message": "There was an error uploading the file"}
    return {"message": "Image downloaded successfully", 'classifier_result':classifier_result}

@app.get("/download-csv/")
async def download_csv():
    with open('data.csv', 'rb') as f:
        csv_bytes = f.read().decode('utf-8')
    csv_io = BytesIO(csv_bytes.encode('utf-8'))
    response = Response(content=csv_io.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename="data.csv"'
    response.headers['Content-Type'] = 'text/csv'
    return response

@app.get("/download-hssp/")
async def download_hssp():
    with open('data.hssp', 'rb') as f:
        hssp_bytes = f.read().decode('utf-16')
    hssp_io = BytesIO(hssp_bytes.encode('utf-16'))
    response = Response(content=hssp_io.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename="data.hssp"'
    response.headers['Content-Type'] = 'text/json'
    return response

@app.get("/download-archive/")
async def download_archive():
    os.makedirs('/temp', exist_ok=True)
    file_paths = ["data.csv", "data.hssp"]
    for file_path in file_paths:
        shutil.copy(file_path, '/temp')
    archive_path = shutil.make_archive("archive", "zip", '/temp')
    shutil.rmtree('/temp')
    return FileResponse(archive_path, media_type="application/zip", filename="archive.zip")

if __name__=='__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)