from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

from PIL import Image

# model_path = "liuhaotian/llava-v1.5-7b"

# tokenizer, model, image_processor, context_len = load_pretrained_model(
#     model_path=model_path,
#     model_base=None,
#     model_name=get_model_name_from_path(model_path)
# )

model_path = "liuhaotian/llava-v1.5-7b"
prompt = "how many cars in the photo? what colors are they? dont include trucks"
# prompt = "do you see red color here?"
# image_file = "https://llava-vl.github.io/static/images/view.jpg"
# img = Image.open("playground/data/eval/vqav2/test2015/COCO_test2015_000000000001.jpg")
img = "playground/data/eval/vqav2/test2015/COCO_test2015_000000000001.jpg"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "load_4bit": True,
    "query": prompt,
    "conv_mode": None,
    "image_file": img,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 128,
})()

eval_model(args)

def count_parameters(model):
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_parameters


# print(f"Number of parameters: {count_parameters(model)}")


# 512 max_new_tokens 6759395328 params FLOPS: 726.798 GFLOPS, MACs: 363.389 GMACs, Params: 6.7594 B
