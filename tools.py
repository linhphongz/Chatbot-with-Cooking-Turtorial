from langchain.tools import BaseTool
from transformers import BlipProcessor, BlipForConditionalGeneration,DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
class ImageCaptionTool(BaseTool):
    name = None
    description = None
    def run(self,image_path):
        img = Image.open(image_path).convert('RGB')
        model_name = "Salesforce/blip-image-captioning-large"
        device = "cpu"

        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        inputs = processor(img, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=25)
        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption
class ObjectDetector(BaseTool):
    name = None
    description = None
    def run(self,image_path):
        image = Image.open(image_path)
        # you can specify the revision tag if you don't want the timm dependency
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
        detections = ''
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections += '[{},{},{},{}]'.format(int(box[0]),int(box[1]),int(box[2]),int(box[3]))
            detections += '{}'.format(model.config.id2label[int(label)])
            detections += '{}\n'.format(float(score))
        return detections