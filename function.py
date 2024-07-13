from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

def get_image_caption(image_path):
    img = Image.open(image_path).convert('RGB')
    model_name = "Salesforce/blip-image-captioning-large"
    device = "cpu"

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    inputs = processor(img, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=25)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


if __name__ == '__main__':
    image_path = "./your_path_file"
    caption = get_image_caption(image_path)