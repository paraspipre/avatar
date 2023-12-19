from pydantic import BaseModel
from typing import List, Tuple


class BaseSDRequest(BaseModel):
    prompt: str ="superman"
    encoded_image: str
    height: int = 512
    width: int=512
    negative_prompt: str = "deformed, nsfw, blurr"
    base_model:str = "stabilityai/stable-diffusion-xl-base-1.0"
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
    # encoded_image: str
    height: int = 512
    width: int=512
    negative_prompt: str = "deformed, nsfw, blurr"
    base_model:str = "stabilityai/stable-diffusion-xl-base-1.0"
    num_inference_steps: int = 30
    guidance_scale:float = 7.5
    num_images_per_prompt:int = 1
