import os
from fastapi import FastAPI, HTTPException
import uvicorn
from pyngrok import ngrok
from fastapi.middleware.cors import CORSMiddleware
from base_request_model import BaseSDRequestHD, BaseSDRequest, BaseSDRequestVideo, BaseSDRequestLogo, BaseSDRequestCanny,BaseSDRequestOpenpose,BaseSDRequestRoop
from generateImage import generateImage,generateLogo, generateOpenpose,generateVideo, generateCanny,generateRoop
from datetime import datetime
import cloudinary
import cloudinary.uploader
import base64
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

@app.post("/generateImageHritik")
async def generate_image(base_request: BaseSDRequestRoop):
    try:
        req_id = datetime.now().strftime("%Y%m%d%H%M%S")
        generated_image_encoded = generateRoop(base_request, req_id)
        

        # Set your Cloudinary credentials
        cloudinary.config( 
            cloud_name = "dwouepph4", 
            api_key = "945814147879561", 
            api_secret = "YJtSjnAwmuni7Rcw25wYiN3pMIs" 
        )

        # Upload the image
        uploadStr = 'data:image/jpeg;base64,' + generated_image_encoded;
        upload_response = cloudinary.uploader.upload(uploadStr)

        # Get the image URL
        image_url = upload_response["secure_url"]

        # Print the image URL
        return {
            "prompt": base_request.prompt,
            "generated_image_encoded": image_url
        }

    except Exception as e:
        print(f"Exception occurred with error as {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generateImage")
async def generate_image(base_request: BaseSDRequest):
    try:
        req_id = datetime.now().strftime("%Y%m%d%H%M%S")
        print(base_request.path)
        if base_request.path == "/text-image":
            generated_image_encoded = generateImage(base_request, req_id)
        elif base_request.path == "/text-logo":
            generated_image_encoded = generateLogo(base_request, req_id)
        elif base_request.path == "/text-video":
            generated_image_encoded = generateVideo(base_request, req_id)

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

        # Set your Cloudinary credentials
        cloudinary.config( 
            cloud_name = "dwouepph4", 
            api_key = "945814147879561", 
            api_secret = "YJtSjnAwmuni7Rcw25wYiN3pMIs" 
        )

        # Upload the image
        uploadStr = 'data:video/mp4;base64,' + generated_image_encoded;
        upload_response = cloudinary.uploader.upload(uploadStr)

        # Get the image URL
        image_url = upload_response["secure_url"]

        return {
            "prompt": base_request.prompt,
            "generated_image_encoded": image_url
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
   uvicorn.run(app,host="0.0.0.0",port=8000)