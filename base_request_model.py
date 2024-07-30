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
    base_model:str = "lykon/dreamshaper-7"
    num_inference_steps: int = 30
    guidance_scale:float = 7.5
    num_images_per_prompt:int = 1

class BaseSDRequestRoopPose(BaseModel):
    prompt: str ="superman"
    encoded_image: str
    pose_image: str
    height: int = 1024
    width: int=1024
    negative_prompt: str = "deformed, blurr"
    base_model:str = "lykon/dreamshaper-7"
    num_inference_steps: int = 30
    guidance_scale:float = 7.5
    num_images_per_prompt:int = 1
