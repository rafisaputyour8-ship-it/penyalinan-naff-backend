import io
import os
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import google.generativeai as genai

# =====================
# APP INFO
# =====================
APP_NAME = "Penyalinan Naff"

PROMPT_OCR = """
Anda adalah OCR komik profesional.

Ekstrak semua teks dari gambar dan susun rapi dengan format:
"": dialog
(): pikiran
[]: narasi
<>: efek suara
ST: teks luar panel

JANGAN beri penjelasan tambahan.
"""

# =====================
# FASTAPI INIT
# =====================
app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

# =====================
# CORE OCR FUNCTION
# =====================
def run_ocr(image_bytes: bytes, api_key: str) -> str:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        image = Image.open(io.BytesIO(image_bytes))
        response = model.generate_content([PROMPT_OCR, image])
        return response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================
# SINGLE IMAGE OCR
# =====================
@app.post("/ocr")
async def ocr_image(
    file: UploadFile = File(...),
    gemini_key: str = Header(...)
):
    image_bytes = await file.read()
    result = run_ocr(image_bytes, gemini_key)
    return {"result": result}

# =====================
# BULK OCR
# =====================
@app.post("/ocr-bulk")
async def ocr_bulk(
    files: list[UploadFile] = File(...),
    gemini_key: str = Header(...)
):
    results = []
    for idx, file in enumerate(files, start=1):
        image_bytes = await file.read()
        text = run_ocr(image_bytes, gemini_key)
        results.append(f"/// HALAMAN {idx} ///\n{text}")

    return {"result": "\n\n".join(results)}
