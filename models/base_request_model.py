from pydantic import BaseModel
from typing import List, Tuple


class BaseSDRequest(BaseModel):
    prompt: str ="superman"
    height: int = 512
    width: int=512
    negative_prompt: str = "deformed, blurr"
    base_model:str = "lykon/dreamshaper-7"
    num_inference_steps: int = 30
    guidance_scale:float = 7.5
    num_images_per_prompt:int = 1
    path : str = "/text-image"

class BaseSDRequestRoop(BaseModel):
    prompt: str ="superman"
    encoded_image: str
    height: int = 1024
    width: int=1024
    negative_prompt: str = "deformed, blurr"
    base_model:str = "JoPmt/Txt2Img_Jggrnt_XL_V7_Pipe"
    num_inference_steps: int = 30
    guidance_scale:float = 7.5
    num_images_per_prompt:int = 1


class BaseSDRequestVideo(BaseModel):
    prompt: str ="superman"
    height: int = 512
    width: int=512
    negative_prompt: str = "deformed, nsfw, blurr"
    base_model:str = "stabilityai/stable-diffusion-xl-base-1.0"
    num_inference_steps: int = 30
    guidance_scale:float = 7.5
    num_images_per_prompt:int = 1


class BaseSDRequestLogo(BaseModel):
    prompt: str ="superman"
    height: int = 512
    width: int=512
    negative_prompt: str = "deformed, nsfw, blurr"
    base_model:str = "stabilityai/stable-diffusion-xl-base-1.0"
    num_inference_steps: int = 30
    guidance_scale:float = 7.5
    num_images_per_prompt:int = 1


class BaseSDRequestCanny(BaseModel):
    prompt: str ="superman"
    encoded_image: str
    height: int = 512
    width: int=512
    negative_prompt: str = "deformed, nsfw, blurr"
    base_model:str = "stabilityai/stable-diffusion-xl-base-1.0"
    num_inference_steps: int = 30
    guidance_scale:float = 7.5
    num_images_per_prompt:int = 1


class BaseSDRequestOpenpose(BaseModel):
    prompt: str ="superman"
    encoded_image: str
    height: int = 512
    width: int=512
    negative_prompt: str = "deformed, nsfw, blurr"
    base_model:str = "stabilityai/stable-diffusion-xl-base-1.0"
    num_inference_steps: int = 30
    guidance_scale:float = 7.5
    num_images_per_prompt:int = 1

class BaseSDRequestLogo(BaseModel):
    prompt: str ="superman"
    height: int = 512
    width: int=512
    negative_prompt: str = "deformed, nsfw, blurr"
    base_model:str = "stabilityai/stable-diffusion-xl-base-1.0"
    num_inference_steps: int = 30
    guidance_scale:float = 7.5
    num_images_per_prompt:int = 1
