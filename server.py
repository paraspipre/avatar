import os
from fastapi import FastAPI, HTTPException
import uvicorn
from models.base_request_model import BaseSDRequest
from pipeline.generateImage import run_generate
from datetime import datetime
from pyngrok import ngrok
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/check")
async def heartbeat():
   return {"status":"alive"}

@app.post("/generateImage")
async def generate_image(base_request: BaseSDRequest):
    try:

        req_id = datetime.now().strftime("%Y%m%d%H%M%S")
        print(req_id)
        # Call the inpainting function
        generated_image_encoded = run_generate(base_request, req_id)

        return {
            "prompt": base_request.prompt,
            "generated_image_encoded": generated_image_encoded
        }

    except Exception as e:
        print(f"Exception occurred with error as {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
   port=8000
   public_url = ngrok.connect(port).public_url
   print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, port))

# Update any base URLs to use the public ngrok URL
   os.environ["BASE_URL"] = public_url
   uvicorn.run(app,host="0.0.0.0",port=8000)