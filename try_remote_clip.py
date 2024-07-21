from huggingface_hub import hf_hub_download
import torch, open_clip
from PIL import Image
from IPython.display import display

# for model_name in ['ViT-B-32']:
#     checkpoint_path = hf_hub_download("chendelong/RemoteCLIP", f"RemoteCLIP-{model_name}.pt", cache_dir="/media/zpp2/PHDD/RemoteCLIP")
#     print(f'{model_name} is downloaded to {checkpoint_path}.')

model_name = 'ViT-B-32' # 'RN50' or 'ViT-B-32' or 'ViT-L-14'
model, _, preprocess = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)

path_to_your_checkpoints = '/media/zpp2/PHDD/RemoteCLIP/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38'

ckpt = torch.load(f"{path_to_your_checkpoints}/RemoteCLIP-{model_name}.pt", map_location="cpu")
message = model.load_state_dict(ckpt)
print(message)
model = model.cuda().eval()
text_queries = [
    "A busy airport with many aireplans.", 
    "Satellite view of Hohai university.", 
    "A building next to a lake.", 
    "Many people in a stadium.", 
    "a cute cat",
    ]
text = tokenizer(text_queries)
image = Image.open("airport.jpg")
# display(image)
print(image.size)
image = preprocess(image).unsqueeze(0)
print(image.shape)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image.cuda())
    print(image_features.shape)
    text_features = model.encode_text(text.cuda())
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]