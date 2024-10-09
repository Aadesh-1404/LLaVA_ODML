import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from datetime import date

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

# import nltk
# from nltk.corpus import stopwords

from time import time

from calflops import calculate_flops

import numpy as np

## find number of flops for each input and take avergae of all inputs
# from fvcore.nn import FlopCountAnalysis, flop_count_table


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(
        self, questions, image_folder, tokenizer, image_processor, model_config
    ):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert("RGB")
        # change resolution of image to test different resolutions
        height, width = image.size
        scale = 24
        # scale = 48
        image = image.resize((height // scale, width // scale))

        image_tensor = process_images([image], self.image_processor, self.model_config)[
            0
        ]

        ## Implement stop words for input text
        # prompt = prompt.split()
        # prompt = ' '.join([word for word in prompt if word.lower() not in stopwords.words('english')])

        # import pdb; pdb.set_trace()

        ## Implement random word removal for input text
        # prompt = prompt.split()
        # prompt = ' '.join([word for word in prompt if np.random.rand() > 0.5])

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(
    questions,
    image_folder,
    tokenizer,
    image_processor,
    model_config,
    batch_size=1,
    num_workers=4,
):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(
        questions, image_folder, tokenizer, image_processor, model_config
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    def count_parameters(model):
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return num_parameters

    num_params = count_parameters(model)

    questions = [
        json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")
    ]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if (
        "plain" in model_name
        and "finetune" not in model_name.lower()
        and "mmtag" not in args.conv_mode
    ):
        args.conv_mode = args.conv_mode + "_mmtag"
        print(
            f"It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}."
        )

    start = time()

    data_loader = create_data_loader(
        questions, args.image_folder, tokenizer, image_processor, model.config
    )

    data_loader_time = time() - start

    inference_time = []
    flops_list = []

    for (input_ids, image_tensor, image_sizes), line in tqdm(
        zip(data_loader, questions), total=len(questions)
    ):
        start_inference = time()

        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device="cuda", non_blocking=True)

        flops, macs, params = calculate_flops(
            model=model,
            input_shape=(1, input_ids.size(1)),
            transformer_tokenizer=tokenizer,
            output_precision=16,
            print_results=False,
        )
        flops_list.append(flops)
        # print(f">>> FLOPS: {flops}, MACs: {macs}, Params: {params}")

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(
                    dtype=torch.float16, device="cuda", non_blocking=True
                ),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": cur_prompt,
                    "text": outputs,
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "metadata": {},
                }
            )
            + "\n"
        )
        # ans_file.flush()

        inference_time.append(time() - start_inference)

    end = time()

    time_elapsed = end - start

    ## if file exists, create a new file with a different name, add time and date to the name
    if os.path.exists(f"{model_name}.txt"):
        model_name = model_name + "_" + str(date.today()) + "_" + str(int(time()))
    # store num_params and time_elapsed in model_name.txt (create if not exists)
    with open(f"{model_name}.txt", "a") as f:
        f.write(
            f"num params: {num_params}, total inference time: {time_elapsed}\n"
        )  # total flops: {sum(flops_list)}
        f.write(f"Data Loader Time: {data_loader_time}\n")
        f.write(
            f"Average Inference Time: {sum(inference_time) / len(inference_time)}\n"
        )
        f.write("\n")
        f.write(f"Inference Time for each batch:\n")
        f.write(f" ".join([str(x) for x in inference_time]))
        f.write("\n")
        f.write("Flops for each batch:\n")
        f.write(" ".join([str(x) for x in flops_list]))
        f.write("\n")
        # extract only integer values from flops_list, it also contains FLops
        flops_list = [int(x.split(" ")) for x in flops_list]
        f.write(f"Average FLOPs: {np.mean(flops_list)}\n")
        # f.write(f"FLOPs for each batch:\n")
        # f.write(" ".join([str(x) for x in flops_list]))

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
