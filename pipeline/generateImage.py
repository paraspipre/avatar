# set up
import torch
from diffusers import AutoPipelineForText2Image
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

from utils.imageUtil import decode_base64_image, encode_image
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



def generate_image(base_request,req_id):
    # Load the pipeline based on the model path in the request
   #  load_pipeline(base_request.base_model)
    print(base_request)
    current_model_path =base_request.base_model
    model = AutoPipelineForText2Image.from_pretrained(
        base_request.base_model,
        torch_dtype=d_type, variant="fp16", use_safetensors=True
    ).to(device)

    # model.load_lora_weights("/content/drive/MyDrive/Harrlogos_v2.0.safetensors", weight_name="Harrlogos_v2.0.safetensors")
    state_dict, network_alphas = model.lora_state_dict(
    "/content/drive/MyDrive/Harrlogos_v2.0.safetensors",
    unet_config=model.unet.config,
    torch_dtype=d_type, variant="fp16", use_safetensors=True,ignore_mismatched_sizes=True
    )
    model.load_lora_into_unet(
    state_dict,
    network_alphas=network_alphas,
    unet=model.unet,
    low_cpu_mem_usage=False
    )

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
    final_image_path = "output.png"
    image.save(final_image_path)
    image.save("output_preview.png")
    # print("roopstart")
    # roop_image_path = get_roop_enhanced_image(user_image_path, final_image_path)
    # print("roopdone")
    # return roop_image_path
    return final_image_path

    

def get_roop_enhanced_image(user_image_path, generated_image_path):
    roop_image_path = "output_roop.png"

    # Get the absolute path of the "roop" directory
    roop_directory = os.path.join(os.path.dirname(__file__), "roop")

    try:
        # Set the working directory to the "roop" directory
        # os.chdir(roop_directory)
        # os.chdir("./roop")
        # Run the subprocess in the "roop" directory
        # subprocess.run(["cd ./roop","&&","python", "run.py", "-s", user_image_path, "-t", generated_image_path, "-o", roop_image_path], check=True)
        command = "cd ./roop && python run.py -s ../{} -t ../{} -o ../{}".format(user_image_path, generated_image_path, roop_image_path)
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        # Reset the working directory to the original directory
        # os.chdir(os.path.dirname(__file__))

        # Raise a ValueError if the subprocess fails
        raise ValueError(f"Roop enhancement failed: {e}")

    finally:
        # Reset the working directory to the original directory (in case of an exception)
        os.chdir(os.path.dirname(__file__))

    return roop_image_path

import os
def delete_image_file(file_path):
    try:
        os.remove(file_path)
        print(f"File {file_path} deleted successfully.")
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def run_generate(base_request,req_id) -> str:
    final_image_path = generate_image(base_request, req_id)
    generated_image_encoded = encode_image("../../"+final_image_path)
    # once get it encoded, delete the file
   #  delete_image_file(final_image_path)
    return generated_image_encoded



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