import os
import shutil
import zipfile
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from bereal_script import reorder_secondary_photos, process_images, create_timelapse

app = FastAPI()

UPLOAD_FOLDER = "/tmp/uploads"
OUTPUT_FOLDER = "/tmp/timelapse_videos"
REORDER_OUTPUT_FOLDER = "/tmp/reorder_output"
ALIGNED_OUTPUT_FOLDER = "/tmp/aligned_output"
PREDICTOR_PATH = "/path/to/shape_predictor_68_face_landmarks.dat"

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), email: str = Form(...)):
    zip_path = os.path.join(UPLOAD_FOLDER, file.filename)
    extract_path = os.path.join(UPLOAD_FOLDER, "extracted")

    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        reorder_secondary_photos(extract_path, "posts.json", REORDER_OUTPUT_FOLDER)
        process_images(REORDER_OUTPUT_FOLDER, ALIGNED_OUTPUT_FOLDER, PREDICTOR_PATH)
        create_timelapse(ALIGNED_OUTPUT_FOLDER, OUTPUT_FOLDER)

        return JSONResponse({"success": True, "message": "Timelapse created!"})
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
