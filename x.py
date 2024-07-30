import requests

API_URL = "https://api-inference.huggingface.co/models/sd-dreambooth-library/homelander"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content
image_bytes = query({
	"inputs": "a towering figure emerges from the shadows at the convention, its godly presence commanding attention. the 35 year old deity stands atop a dais of black stone, its piercing blue eyes glowing like lanterns in the dark. the air is heavy with the scent of smoke and ozone as thunder rumbles outside, echoing the turmoil within. a tattered cloak billows behind it, embroidered with ancient runes that seem to writhe and twist like living serpents. the god's face is a mask of unyielding fury, its features chiseled from granite and adorned with horns that curve like scimitars. at its feet lies a altar of black marble, upon which rest three gleaming obsidian orbs that seem to pulse with an otherworldly energy. as the convention attendees watch in awed terror, the god's gaze falls upon them, as if sizing them up for some unknown purpose. the atmosphere is heavy with foreboding, the very fabric of reality seeming to warp and distort around this divine presence., <lora:sdxl_lightning_8step_lora.safetensors:0.6>",
})
# You can access the image with PIL.Image for example
import io
from PIL import Image
image = Image.open(io.BytesIO(image_bytes))
image.save("image.png")