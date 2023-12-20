import os
from fastapi import FastAPI, HTTPException
import uvicorn
from models.base_request_model import BaseSDRequest, BaseSDRequestVideo, BaseSDRequestLogo, BaseSDRequestCanny,BaseSDRequestOpenpose,BaseSDRequestRoop
from pipeline.generateImage import generateImage,generateLogo, generateOpenpose,generateVideo, generateCanny,generateRoop
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
async def check():
   return True

@app.post("/generateImage")
async def generate_image(base_request: BaseSDRequest):
    try:
        req_id = datetime.now().strftime("%Y%m%d%H%M%S")

        generated_image_encoded = generateImage(base_request, req_id)

        return {
            "prompt": base_request.prompt,
            "generated_image_encoded": generated_image_encoded
        }

    except Exception as e:
        print(f"Exception occurred with error as {e}")
        raise HTTPException(status_code=500, detail=str(e))

 
@app.post("/generateRoop")
async def generate_roop(base_request: BaseSDRequestRoop):
    try:
        req_id = datetime.now().strftime("%Y%m%d%H%M%S")

        generated_image_encoded = generateRoop(base_request, req_id)

        return {
            "prompt": base_request.prompt,
            "generated_image_encoded": generated_image_encoded
        }

    except Exception as e:
        print(f"Exception occurred with error as {e}")
        raise HTTPException(status_code=500, detail=str(e))

    
@app.post("/generateVideo")
async def generate_video(base_request: BaseSDRequestVideo):
    try:

        req_id = datetime.now().strftime("%Y%m%d%H%M%S")

        generated_image_encoded = generateVideo(base_request, req_id)

        return {
            "prompt": base_request.prompt,
            "generated_image_encoded": generated_image_encoded
        }

    except Exception as e:
        print(f"Exception occurred with error as {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/generateLogo")
async def generate_logo(base_request: BaseSDRequestLogo):
    try:

        req_id = datetime.now().strftime("%Y%m%d%H%M%S")
        # print(req_id)
        # Call the inpainting function
        generated_image_encoded = generateLogo(base_request, req_id)

        return {
            "prompt": base_request.prompt,
            "generated_image_encoded": generated_image_encoded
        }

    except Exception as e:
        print(f"Exception occurred with error as {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generateCanny")
async def generate_canny(base_request: BaseSDRequestCanny):
    try:

        req_id = datetime.now().strftime("%Y%m%d%H%M%S")
        # print(req_id)
        # Call the inpainting function
        generated_image_encoded = generateCanny(base_request, req_id)

        return {
            "prompt": base_request.prompt,
            "generated_image_encoded": generated_image_encoded
        }

    except Exception as e:
        print(f"Exception occurred with error as {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generateOpenpose")
async def generate_openpose(base_request: BaseSDRequestOpenpose):
    try:

        req_id = datetime.now().strftime("%Y%m%d%H%M%S")
        # print(req_id)
        # Call the inpainting function
        generated_image_encoded = generateOpenpose(base_request, req_id)

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