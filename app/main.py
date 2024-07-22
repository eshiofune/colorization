import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from PIL import Image
import io
import utils

CHECKPOINT_PATHS = {"1": "checkpoints/check_18k_1.pth", "4": "checkpoints/check_18k_4.pth"}
MODELS = utils.load_models(CHECKPOINT_PATHS)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def get():
    return FileResponse('templates/index.html')


@app.post("/colorize/")
async def colorize_image(modelId: str = Form(...), file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        model = MODELS[modelId]
        print(f"Using model: {CHECKPOINT_PATHS[modelId]}")
        input_tensor, original_size = utils.preprocess_image(image)

        with torch.no_grad():
            output_tensor = model.G_net(input_tensor)

        colored_image = utils.postprocess_output(output_tensor, input_tensor, original_size)

        buf = io.BytesIO()
        colored_image.save(buf, format="PNG")
        buf.seek(0)
        print("Done")
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        print("An error occurred:" + e.with_traceback(e.__traceback__))
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
