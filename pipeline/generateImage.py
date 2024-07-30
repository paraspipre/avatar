# set up
import torch
from diffusers import StableDiffusionLatentUpscalePipeline, StableDiffusionXLImg2ImgPipeline, AutoPipelineForText2Image, MotionAdapter, AnimateDiffPipeline, DDIMScheduler,StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL,StableDiffusionControlNetPipeline,UniPCMultistepScheduler,StableDiffusionXLPipeline
from diffusers.utils import export_to_video,load_image, make_image_grid
from PIL import Image
import cv2
import numpy as np
from compel import Compel, ReturnedEmbeddingsType

from controlnet_aux import OpenposeDetector

# from utils.control_net_utils import CONTROLNET_MAPPING
# from IP_Adapter.ip_adapter.ip_adapter import IPAdapter, IPAdapterPlus

# vae_model_path = "stabilityai/sd-vae-ft-mse"
# image_encoder_path = "IP-Adapter/models/image_encoder/"
# ip_ckpt = "IP-Adapter/models/ip-adapter-plus-face_sd15.bin"

device = "cuda"
d_type = torch.float16
torch.cuda.empty_cache()


# def setup_pipeline(base_model_path: str = "Yntec/epiCPhotoGasm"):

#     control_type = "pose"
#     controlnet = ControlNetModel.from_pretrained(CONTROLNET_MAPPING[control_type]["model_id"], torch_dtype=d_type).to(device)

#     pipe_control_net = StableDiffusionControlNetPipeline.from_pretrained(base_model_path,
#                                                       controlnet=controlnet,
#                                                       torch_dtype=d_type,
#                                                       safety_checker=None,
#                                                       controlnet_conditioning_scale=0.8).to(device)

#     # load ip-adapter
#     print("DEBUG: loading IP adapter ")
#     ip_model = IPAdapterPlus(pipe_control_net, image_encoder_path, ip_ckpt, device="cuda", num_tokens=16)


#     return ip_model

from utils.imageUtil import decode_base64_image, encode_image, delete_image_file
from models.base_request_model import BaseSDRequest

import subprocess
import os


# Define a global variable to track the loaded model path
current_model_path = None
model = None


# pipe_inpaint = setup_pipeline(base_model_path = "stabilityai/stable-diffusion-xl-base-1.0")
# def load_pipeline(model_path):
#     global current_model_path, ip_model
#     if current_model_path != model_path:
#         # Load the pipeline only if the model path has changed
#         ip_model = setup_pipeline(base_model_path=model_path)
#         current_model_path = model_path
#         print(f"\nChanging model to {model_path}\n")



def generateImage(base_request,req_id):
    print(base_request)
    model = AutoPipelineForText2Image.from_pretrained(
        base_request.base_model,
        torch_dtype=d_type, use_safetensors=True
    ).to(device)

    # model.load_lora_weights("/content/drive/MyDrive/Harrlogos_v2.0.safetensors", weight_name="Harrlogos_v2.0.safetensors")
    # state_dict, network_alphas = model.lora_state_dict(
    # "/content/drive/MyDrive/Harrlogos_v2.0.safetensors",
    # unet_config=model.unet.config,
    # torch_dtype=d_type, variant="fp16", use_safetensors=True,ignore_mismatched_sizes=True
    # )
    # model.load_lora_into_unet(
    # state_dict,
    # network_alphas=network_alphas,
    # unet=model.unet,
    # low_cpu_mem_usage=False,
    # # ignore_mismatched_sizes=True
    # )

    # Decode the base64-encoded image
   #  control_net_image = decode_base64_image(base_request.encoded_control_net_image)
   #  control_image = CONTROLNET_MAPPING[base_request.control_type]["hinter"](control_net_image)

    # sd_pipe.height = base_request.height
    # sd_pipe.width = base_request.width
    # sd_pipe.image = control_image
    # sd_pipe.num_images_per_prompt = 1
    # sd_pipe.guidance_scale = base_request.guidance_scale
    # sd_pipe.controlnet_conditioning_scale = base_request.controlnet_conditioning_scale
    # sd_pipe.num_inference_steps = base_request.num_inference_steps

    # user_image = decode_base64_image(base_request.encoded_image)
    # user_image_path = "user_image.png"
    # user_image.save(user_image_path)

    import random
    random_seed = random.randint(1, 1000000)

    image = model(prompt=base_request.prompt,
                  negative_prompt=base_request.negative_prompt,
                  seed=random_seed,
                               width=base_request.width,
                               height=base_request.height,
                               num_samples=1,
                               ).images[0]
    final_image_path = "output.png" + req_id + ".png"
    image.save(final_image_path)
    # print("roopstart")
    # roop_image_path = get_roop_enhanced_image(user_image_path, final_image_path)
    # print("roopdone")
    # return roop_image_path
    generated_image_encoded = encode_image(final_image_path)
    # once get it encoded, delete the file
   #  delete_image_file(final_image_path)
    return generated_image_encoded

def generateRoop(base_request,req_id):
# print(base_request)
    pipeline = AutoPipelineForText2Image.from_pretrained(
        base_request.base_model,
        torch_dtype=d_type, variant="fp16", use_safetensors=True
    ).to(device)
    compel = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
    conditioning = compel.build_conditioning_tensor(base_request.prompt)


    upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16).to("cuda")


    user_image = decode_base64_image(base_request.encoded_image)
    user_image_path = "user_image" + req_id + ".png"
    user_image.save(user_image_path)

    import random
    random_seed = random.randint(1, 1000000)

    image = pipeline(prompt_embeds=conditioning, 
                  negative_prompt=base_request.negative_prompt,
                  seed=random_seed,
                               width=base_request.width,
                               height=base_request.height,
                               num_samples=1,
                               output_type="latent"
                               ).images[0]
    
    
    upscaled_image = pipeline(prompt_embeds=conditioning, image=image).images[0]
    final_image_path = "output" + req_id + ".png"
    upscaled_image.save(final_image_path)
    # print("roopstart")
    roop_image_path = get_roop_enhanced_image(user_image_path, final_image_path,req_id)
    # print("roopdone")
    # return roop_image_path
    generated_image_encoded = encode_image(roop_image_path)
    # once get it encoded, delete the file
    # delete_image_file(final_image_path)
    # delete_image_file(user_image_path)
    # delete_image_file(final_image_path)
    return generated_image_encoded

def get_roop_enhanced_image(user_image_path, generated_image_path,req_id):
    roop_image_path = "output_roop" + req_id+ ".jpg"

    try:
        subprocess.run("pwd", shell=True, check=True)
        command = "cd {} && python run.py -s ../{} -t ../{} -o ../{} ".format("./roop",user_image_path, generated_image_path, roop_image_path)
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        # Reset the working directory to the original directory
        # os.chdir(os.path.dirname(__file__))

        # Raise a ValueError if the subprocess fails
        raise ValueError(f"Roop enhancement failed: {e}")

    finally:
        # Reset the working directory to the original directory (in case of an exception)
        # os.chdir(os.path.dirname(__file__))
        pass
        
    return roop_image_path

def get_roop_enhanced_video(user_image_path, generated_image_path,req_id):
    roop_image_path = "output_roop" + req_id+ ".mp4"

    try:
        subprocess.run("pwd", shell=True, check=True)
        command = "cd {} && python run.py --frame-processors face_swapper -s ../{} -t ../{} -o ../{} --headless".format("./facefusion",user_image_path, generated_image_path, roop_image_path)
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        # Reset the working directory to the original directory
        # os.chdir(os.path.dirname(__file__))

        # Raise a ValueError if the subprocess fails
        raise ValueError(f"Roop enhancement failed: {e}")

    finally:
        # Reset the working directory to the original directory (in case of an exception)
        # os.chdir(os.path.dirname(__file__))
        pass
        
    return roop_image_path



def generateVideo(base_request,req_id):

    # Load the motion adapter
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
    # load SD 1.5 based finetuned model
    model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
    pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter)
    scheduler = DDIMScheduler.from_pretrained(
        model_id, subfolder="scheduler", clip_sample=False, timestep_spacing="linspace", steps_offset=1
    )
    pipe.scheduler = scheduler

    # enable memory savings
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()

    user_image = decode_base64_image(base_request.encoded_image)
    user_image_path = "user_image" + req_id + ".jpg"
    user_image.save(user_image_path)

    output = pipe(
        # prompt=(
        #     "masterpiece, bestquality, highlydetailed, ultradetailed, sunset, "
        #     "orange sky, warm lighting, fishing boats, ocean waves seagulls, "
        #     "rippling water, wharf, silhouette, serene atmosphere, dusk, evening glow, "
        #     "golden hour, coastal landscape, seaside scenery"
        # ),
        prompt=base_request.prompt,
        negative_prompt=base_request.negative_prompt,
        num_frames=16,
        guidance_scale=7.5,
        num_inference_steps=25,
    )
    frames = output.frames[0]
    user_video_path = "user_video" + req_id + ".mp4"
    export_to_video(frames, user_video_path)

    # final_image_path = "output.png"
    # image.save(final_image_path)
    # image.save("output_preview.png")
    # print("roopstart")
    
    roop_image_path = get_roop_enhanced_video(user_image_path, user_video_path,req_id)
    # print("roopdone")
    # return roop_image_path
    generated_image_encoded = encode_image(roop_image_path)
    # once get it encoded, delete the file
   #  delete_image_file(final_image_path)
    return generated_image_encoded
    # return "okay"

import re
from PIL import Image
import base64
from io import BytesIO

def generateCanny(base_request,req_id):

    user_image = decode_base64_image(base_request.encoded_image)
    user_image_path = "user_image.png"
    user_image.save(user_image_path)
    image = np.array(user_image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    canny_image.save("cannyImage.png")

    # controlnet = ControlNetModel.from_pretrained(
    #     "diffusers/controlnet-canny-sdxl-1.0",
    #     torch_dtype=torch.float16,
    #     use_safetensors=True
    # )
  
    # vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)
  
    # pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-base-1.0",
    #     controlnet=controlnet,
    #     vae=vae,
    #     torch_dtype=torch.float16,
    #     use_safetensors=True
    # )
    
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", use_safetensors=True)
    pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",safety_checker=None, controlnet=controlnet, use_safetensors=True).to("cuda")

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    
    # prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
    # negative_prompt = 'low quality, bad quality, sketches'

    image = pipe(
        prompt=base_request.prompt,
        negative_prompt=base_request.negative_prompt,
        image=canny_image,
        guidance_scale=3.0,
        guess_mode=True,
    ).images[0]
    image.save("canny_output.png")

    generated_image_encoded = encode_image("canny_output.png")
    # once get it encoded, delete the file
   #  delete_image_file(final_image_path)
    return generated_image_encoded

def generateOpenpose(base_request,req_id):

    user_image = decode_base64_image(base_request.encoded_image)
    user_image_path = "user_image.png"
    user_image.save(user_image_path)

    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    openpose_image = openpose(user_image)
    openpose_image.save("openpose.png")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose", use_safetensors=True
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",safety_checker=None, controlnet=controlnet, use_safetensors=True).to("cuda")

    # vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)
    # pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16, use_safetensors=True
    # )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    
    
    image = pipe(
        prompt=base_request.prompt,
        negative_prompt=base_request.negative_prompt,
        image=openpose_image,
        guidance_scale=3.0,
        guess_mode=True,
    ).images[0]
    image.save("openpose_output.png")

    generated_image_encoded = encode_image("openpose_output.png")
    # once get it encoded, delete the file
   #  delete_image_file(final_image_path)
    return generated_image_encoded


def generateLogo(base_request,req_id):
    # Load the pipeline based on the model path in the request
   #  load_pipeline(base_request.base_model)
    print(base_request)
    current_model_path =base_request.base_model
    model = AutoPipelineForText2Image.from_pretrained(
        base_request.base_model,
        torch_dtype=d_type, variant="fp16", use_safetensors=True
    ).to(device)

    model.load_lora_weights("/kaggle/input/harrlogos-v2-0-safetensors", weight_name="Harrlogos_v2.0.safetensors")
    # state_dict, network_alphas = model.lora_state_dict(
    # "/content/drive/MyDrive/Harrlogos_v2.0.safetensors",
    # unet_config=model.unet.config,
    # torch_dtype=d_type, variant="fp16", use_safetensors=True,ignore_mismatched_sizes=True
    # )
    # model.load_lora_into_unet(
    # state_dict,
    # network_alphas=network_alphas,
    # unet=model.unet,
    # low_cpu_mem_usage=False,
    # # ignore_mismatched_sizes=True
    # )

    # Decode the base64-encoded image
   #  control_net_image = decode_base64_image(base_request.encoded_control_net_image)
   #  control_image = CONTROLNET_MAPPING[base_request.control_type]["hinter"](control_net_image)

    # sd_pipe.height = base_request.height
    # sd_pipe.width = base_request.width
    # sd_pipe.image = control_image
    # sd_pipe.num_images_per_prompt = 1
    # sd_pipe.guidance_scale = base_request.guidance_scale
    # sd_pipe.controlnet_conditioning_scale = base_request.controlnet_conditioning_scale
    # sd_pipe.num_inference_steps = base_request.num_inference_steps

    user_image = decode_base64_image(base_request.encoded_image)
    user_image_path = "user_image" + req_id + ".png"
    user_image.save(user_image_path)

    import random
    random_seed = random.randint(1, 1000000)

    image = model(prompt=base_request.prompt,
                  negative_prompt=base_request.negative_prompt,
                  seed=random_seed,
                               width=base_request.width,
                               height=base_request.height,
                               num_samples=1,
                               ).images[0]
    final_image_path = "output.png" + req_id + ".png"
    image.save(final_image_path)
    image.save("output_preview" + req_id + ".png")
    # print("roopstart")
    # roop_image_path = get_roop_enhanced_image(user_image_path, final_image_path)
    # print("roopdone")
    # return roop_image_path
    subprocess.run("pwd", shell=True, check=True)
    generated_image_encoded = encode_image(final_image_path)
    # once get it encoded, delete the file
   #  delete_image_file(final_image_path)
    return generated_image_encoded




# def run_generate(base_request,req_id) -> str:
#     final_image_path = generate_image(base_request, req_id)
#     generated_image_encoded = encode_image(final_image_path)
#     # once get it encoded, delete the file
#    #  delete_image_file(final_image_path)
#     return generated_image_encoded



# def main():
#     # generate mask
#     user_image_path = "../assets/sample_user_image.png"
#     prompt = "deadpool shooting with guns"
#     # Create an instance of InpaintRequest
#     control_type = "pose"
#     control_net_image_path = "../assets/poses/pose (1).jpg"

#     from utils.image_utils import encode_image

#     encoded_ip_image = encode_image(user_image_path)
#     encoded_control_net_image = encode_image(control_net_image_path)
#     request = BaseSDRequest(prompt=prompt,
#                           control_type=control_type,
#                           encoded_ip_image=encoded_ip_image,
#                           encoded_control_net_image=encoded_control_net_image,
#                           height=512,
#                           width=512)

#     # Call the run_inpaint function with the request
#     generate_image(request)

# if __name__ == "__main__":
#     main()