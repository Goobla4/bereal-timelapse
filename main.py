import os
import shutil
import zipfile
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pathlib import Path
import smtplib
from email.message import EmailMessage
from BeRealCompleteProcessScript import reorder_secondary_photos, process_images, create_timelapse

app = FastAPI()

UPLOAD_FOLDER = "/tmp/uploads"
OUTPUT_FOLDER = "/tmp/timelapse_videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), email: str = Form(...)):
    """Handle the uploaded ZIP and send the timelapse video by email."""
    zip_path = os.path.join(UPLOAD_FOLDER, file.filename)
    extract_path = os.path.join(UPLOAD_FOLDER, "extracted")

    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        video_path = os.path.join(OUTPUT_FOLDER, "timelapse.mp4")
        reorder_secondary_photos(extract_path)
        process_images()
        create_timelapse(video_path)

        send_email_with_attachment(email, video_path)
        return JSONResponse({"success": True, "message": "Video sent!"})
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)})

def send_email_with_attachment(receiver_email, file_path):
    msg = EmailMessage()
    msg["Subject"] = "Your BeReal Timelapse Video"
    msg["From"] = os.getenv("EMAIL_USER")
    msg["To"] = receiver_email

    with open(file_path, "rb") as f:
        msg.add_attachment(f.read(), maintype="video", subtype="mp4", filename="timelapse.mp4")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"))
        smtp.send_message(msg)
