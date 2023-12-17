import requests
import os
from dotenv import load_dotenv
from utils.imageUtil import decode_base64_image, encode_image
from models.base_request_model import BaseSDRequest
# Load environment variables from the .env file
load_dotenv()

# Base API URL
BASE_URL = os.getenv("BASE_URL") or "https://f1fd-3-135-152-169.ngrok-free.app"

def send_api_request(base_request: BaseSDRequest):
    api_url = f"{BASE_URL}/generateImage"

    data = {
      #   "encoded_ip_image": base_request.encoded_ip_image,
      #   "encoded_control_net_image": base_request.encoded_control_net_image,
      #   "control_type": base_request.control_type,
        "prompt": base_request.prompt,
        "height": base_request.height,
        "width": base_request.width
    }

    try:
        response = requests.post(api_url, json=data)

        if response.status_code == 200:
            response_data = response.json()

            generated_image_encoded = response_data.get("generated_image_encoded")
            prompt = response_data.get("prompt")

            # Decode and save the generated image
            if generated_image_encoded:
                generated_image_pil = decode_base64_image(generated_image_encoded)
                generated_image_pil.save("server_output.png")

                return generated_image_pil, prompt
            else:
                print("Generated image not found in the response.")
                return None
        else:
            print(f"API request failed with status code {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request Exception: {e}")
        return None

def main():
    # generate mask
    user_image_path = "assets/sample_user_image.png"
    prompt = "deadpool shooting with eyes"
   #  # Create an instance of InpaintRequest
   #  control_type = "pose"
   #  control_net_image_path = "assets/poses/pose (1).jpg"

    from utils.imageUtil import encode_image

   #  encoded_ip_image = encode_image(user_image_path)
   #  encoded_control_net_image = encode_image(control_net_image_path)
    request = BaseSDRequest(prompt=prompt,
                           #  control_type=control_type,
                           #  encoded_ip_image=encoded_ip_image,
                           #  encoded_control_net_image=encoded_control_net_image,
                            height=512,
                            width=512)

    result = send_api_request(request)
    if result:
        generated_image_pil, prompt = result
        print(f"Prompt: {prompt}")


if __name__ == "__main__":
    main()
