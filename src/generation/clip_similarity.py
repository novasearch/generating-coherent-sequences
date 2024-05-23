from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, CLIPTextModel, logging
from typing import List
from torch.nn import CosineSimilarity
import torch

logging.set_verbosity_error()


MODEL_ID = "openai/clip-vit-base-patch32"

processor = CLIPProcessor.from_pretrained(MODEL_ID)
model = CLIPModel.from_pretrained(MODEL_ID).to("cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
text_encoder = CLIPTextModel.from_pretrained(MODEL_ID).to("cpu")

cossim = CosineSimilarity(dim=0)


def text_image_similarity(text: str, image):
    input_encoding = processor(text=text, 
                        images=image,
                        return_tensors="pt", 
                        padding=True,
                        truncation=True)
    
    outputs = model(**input_encoding)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score

    return logits_per_image.detach().numpy()[0]


def text_text_similarity(text_a, text_b):
    text_inputs = tokenizer([text_a, text_b], padding=True, truncation=True, return_tensors="pt")
    text_features = model.get_text_features(**text_inputs)

    return cossim(text_features[0], text_features[1]).item()


if __name__ == "__main__":
    from PIL import Image
    import requests

    # a = text_text_similarity("Hello", "Hello")
    # b = text_text_similarity("A dog", "A wolf")
    # c = text_text_similarity("dog", "cat")
    
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    e = text_image_similarity(["A photo of a cat", "A photo of a dog"], image)
    f = text_image_similarity(["A photo of a cat", "A photo of a dog"], [image, image])

    g = text_image_similarity("A text", image)

    print(e, f, g)