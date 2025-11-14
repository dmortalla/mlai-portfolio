from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load model once
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

def caption_image(file):
    """Generate a caption for the uploaded image."""
    image = Image.open(file).convert("RGB")
    inputs = processor(image, return_tensors="pt")

    output = model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption
