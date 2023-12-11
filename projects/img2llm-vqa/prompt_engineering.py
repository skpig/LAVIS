from copy import deepcopy
import sys
import re
import json
import torch
import os
import random
from tqdm import tqdm
import requests
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

from lavis.common.gradcam import getAttMap
from lavis.models import load_model_and_preprocess

from utils import load_aokvqa, soft_acc, visualize_chat_prompt, batch_chat_prompt, chat_completion 

MODEL_DIR = "/data2/pretrain/"
OUTPUT_DIR = "/data1/huangbz/LAVIS/"

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
def load_model(model_selection):
    model = AutoModelForCausalLM.from_pretrained(model_selection,
                                                 torch_dtype=torch.bfloat16) # OPTIMIZATION: Possible to use bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_selection, 
                                                padding_side='left',
                                                use_fast=False)
    return model,tokenizer

def gradcam_visualization(raw_image, samples):
    # Gradcam visualisation
    dst_w = 720
    w, h = raw_image.size
    scaling_factor = dst_w / w

    resized_img = raw_image.resize((int(w * scaling_factor), int(h * scaling_factor)))
    norm_img = np.float32(resized_img) / 255 # 对亮度进行归一化
    gradcam = samples['gradcams'][0].reshape(24,24).numpy() # [576] -> [24, 24]， 获得最后一层feature的gradcam

    avg_gradcam = getAttMap(norm_img, gradcam, blur=True)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(avg_gradcam)
    ax.set_yticks([])
    ax.set_xticks([])

    fig.savefig('gradcam.png')

def prompt_generation(batch_data, device, model, vis_processors, txt_processors, num_captions):
    """
    Args:
        batch_data: a list of dicts, each dict contains the following keys:
            - image_path: str
            - question: str
            - answer: str
    """

    # ### Preprocess image and text inputs
    # 预处理图片和文本
    image_lst = []
    for item in batch_data:
        image = Image.open(item['image_path'])
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_lst.append(image)

    images = torch.concat([vis_processors["eval"](image).unsqueeze(0) for image in image_lst], dim=0).to(device) # BLIP image processor处理后，图片大小 [bsz, 3, 384, 384]
    questions = [txt_processors["eval"](item['question']) for item in batch_data] # a string
    samples = {"image": images, "text_input": questions}

    # #### 1. Image-Question Matching 
    # Compute the relevancy score of image patches with respect to the question using GradCAM
    # 计算GradCAM
    samples = model.forward_itm(samples=samples) # a dict of {image: [bsz, 3, H, W], text_input: a list of string, gradcams: [bsz, 576=24*24]}

    #DEBUG
    # gradcam_visualization(image_lst[0], samples)

    # #### 2. Image Captioning
    # Generate question-guided captions based on the relevancy score
    samples = model.forward_cap(samples=samples, num_captions=num_captions, num_patches=20) # add a field 'captions': a list of bsz lists of strings
    # print('Examples of question-guided captions: ')
    # print(samples['captions'][0][:5])


    # # #### 3. Question Generation
    # # Generate synthetic questions using the captions
    # with torch.no_grad():
    #     samples = model.forward_qa_generation(samples)
    # # print('Sample Question: {} \nSample Answer: {}'.format(samples['questions'][:5], samples['answers'][:5]))


    # #### 4. Prompt Construction
    # Prepare the prompts for LLM

    # ### Generate answer by calling `predict_answers()` directly
    # add both contexts, (question, answer) exemplars as prompt
    # samples = model.prompts_construction(samples) 

    del samples['image']
    samples['gradcams'] = samples['gradcams'].cpu()
    # release gradients
    model.zero_grad(set_to_none=True)

    rtn = ['.'.join(samples['captions'][i]) for i in range(len(samples['captions']))]
    return rtn

def answer_generation(batch_history):
    preds = [
        chat_completion(history=history, max_tokens=200, temperature=0.)
        for history in batch_history
    ]
    for pred, history in zip(preds, batch_history):
        history.append({'role': 'assistant', 'content': pred})

    return batch_history, preds

def prompt_generation_worker(sample):
    device = torch.device(f"cuda:1")
    
    # ### Load Img2Prompt-VQA model
    model, vis_processors, txt_processors = load_model_and_preprocess(name="img2prompt_vqa", model_type="base", is_eval=True, device=device)

    cur_caption = prompt_generation(sample, device, model, vis_processors, txt_processors)
    torch.cuda.empty_cache()

    return cur_caption

def answer_generation_worker(model_name, sample):
    device = torch.device(f"cuda:1")

    model,tokenizer = load_model(f'{MODEL_DIR}/{model_name}')
    model = model.to(device)

    with torch.no_grad():
        cur_responese = answer_generation(model, tokenizer, [sample], device)

    return cur_responese

def main(model_name='facebook/opt-6.7b', dataset_name='aokvqa', split='val', bsz=2, device=0, stage=None, num_parallel_workers=1):

    if stage == 'prompt_generation':
        # load dataset
        if dataset_name == 'aokvqa':
            dataset = load_aokvqa('./datasets/aokvqa', split)
        else:
            raise NotImplementedError
        length = len(dataset)
        # split_length = (length + num_parallel_workers - 1) // num_parallel_workers

        rtn = prompt_generation_worker(dataset_name, split, device, dataset, bsz) # DEBUG
        # with mp.Pool(num_parallel_workers) as pool:
        #     rtn = pool.starmap(prompt_generation_worker, 
        #                     [(i, 
        #                         dataset[i * split_length: (i+1) * split_length], 
        #                         bsz) 
        #                         for i in range(num_parallel_workers)])
        # rtn = [i for j in rtn for i in j]

        torch.save(rtn, os.path.join(OUTPUT_DIR, f"{dataset_name}_{split}.pt"))
    elif stage == 'answer_generation':
        samples = torch.load(os.path.join(OUTPUT_DIR, f"{dataset_name}_{split}.pt"))
        rtn = answer_generation_worker(model_name, dataset_name, split, device, samples, bsz)

        if dataset_name == 'aokvqa':
            dataset = load_aokvqa('./datasets/aokvqa', split)
        else:
            raise NotImplementedError
        
        acc = soft_acc(dataset, rtn)
        print(f"Accuracy: {sum(acc)/len(acc)}")

        # Visualize error case
        error_cases = []
        for a, s, d in zip(acc, rtn, dataset):
            if a == 0:
                error_cases.append({
                    'question': s['text_input'],
                    'golden_answer': d.get('direct_answers', None),
                    'pred_answer': s['pred'],
                    'image_path': d['image_path'],
                    'prompt': s['Img2Prompts']
                })
        with open('error_cases.json', 'w') as f:
            json.dump(error_cases, f, indent=4)

def error_cases_analysis(error_case):

    # print("========================================")
    # print("============Question====================")
    # print("============Image: see image.png========")
    # os.system(f"cp {error_case['image_path']} error_image.png")
    # print("============Chat History================")

    # TODO: design the prompt history
    # see utils.prompt() for details
    history = [
{
            'role': 'system',
            'content': """
You are required to answer several Visual-Question-Answering queries.
For each query I will provide you with some captions of a image and a question about the image, presented in json format. 
{
"QueryID" : # the id of current query, for example "1" means the first query.
"Question" : # the image-related question you need to answer
"Caption" : # a list of captions of the image. Note that the captions might be conflict and not complete. You can query for more information and apply more rational reasoning. 
}
Answer the question step-by-step according to given captions. 
Your should give your answer in json format. The requirements are as follows:
{
"QueryID": # the id of current query, for example "1" means the first query.
"TurnID": # the number of rounds of the current query.
"Reason": # your reasoning process to answer the question step-by-step.
"Action": # following the "Reason" content, choose what to do next from the following two options:
    {
    "Search": # ask for more captions about a specific object you want to look up in the image. keep the object brief and short.
    "Answer": # if you are confident enough, give an answer in a brief and short manner.
    }
}
"""
        },
        {
            'role': 'user',
            'content': """{
    "QueryID" : 1,
    "Question" : "What item is in front of the fence which is yellow and can be used to fire fighting?"
    "Caption" : ["a fence of picket white boards with a gate", "the house is fenced in in front of a white picketed fence", "a white picket with pink roses in front of it"],
}"""
        },
        {
            'role': 'assistant',
            'content': """{
    "QueryID" : 1,
    "TurnID" : 0,
    "Reason" : "There are pink roses in front of the fence, but the information about the yellow thing is missing. The answer to the question is not present in the captions.",
    "Action" : {
        "Search" : "object in front of the fence"
    }
}"""
        },
        {
            'role': 'user',
            'content': """{
    "QueryID" : 1,
    "Question" : "What item is in front of the fence which is yellow and can be used to fire fighting?"
    "Caption" : ["a fire hydrant in front of a white fence","There's a bush of pink roses in front of the fence"],
}"""
        },
        {
            'role': 'assistant',
            'content': """{
    "QueryID" : 1,
    "TurnID" : 1,
    "Reason" : "There are many objects in the picture in front of the white fence, including roses and fire hydrants. Need to identify the yellow object in the question",
    "Action" : {
        "Search" : "yellow object in front of the fence"
    }
}"""
        },
        {
            'role': 'user',
            'content': """{
    "QueryID" : 1,
    "Question" : "What item is in front of the fence which is yellow and can be used to fire fighting?"
    "Caption" : ["a yellow fire hydrant in front of a white fence","There's a yellow cylinder in front of a white fence"],
}"""
        },
        {
            'role': 'assistant',
            'content': """{
    "QueryID" : 1,
    "TurnID" : 2,
    "Reason" : "The yellow fire hydrant is in front of the white fence, while it can also be used to put out fires.",
    "Action" : {
        "Answer" : "fire hydrant"
    }
}"""
        },
        {
            'role': 'user',
            'content': """{
    "QueryID" : 2,
    "Question" : "What is the jersey number of the player shooting the three?"
    "Caption" : ["A Lakers player, donning the jersey number 23, is in the middle of taking a three-point shot while being closely guarded by opposing players","The crowd holds its breath in anticipation while the Pacers' defenders try to block his every move" ,"A player, identified by his jersey number 25, launches a three-point attempt", "The basketball court is alive with energy as the Los Angeles Lakers take on the Indiana Pacers." ],
}"""
        },
        {
            'role': 'assistant',
            'content': """{
    "QueryID" : 2,
    "TurnID" : 0,
    "Reason" : "In order to know the jersey number. Need to find the person who shot three.",
    "Action" : {
        "Search" : "the person who shot three"
    }
}"""
        },
        {
            'role': 'user',
            'content': """{
    "QueryID" : 2,
    "Question" : "What is the jersey number of the player shooting the three?"
    "Caption" : ["A player in a yellow is shooting three."],
}"""
        },
        {
            'role': 'assistant',
            'content': """{
    "QueryID" : 2,
    "TurnID" : 1,
    "Reason" : "The player who is shooting three is wearing yellow. Maybe he is from the Laker.",
    "Action" : {
        "Search" : "jersey number of the Laker player in yellow who shot the ball"
    }
}"""
        },
        {
            'role': 'user',
            'content': """{
    "QueryID" : 2,
    "Question" : "What is the jersey number of the player shooting the three?"
    "Caption" : ["A player in a yellow No. 23 jersey is shooting three."],
}"""
        },
        {
            'role': 'assistant',
            'content': """{
    "QueryID" : 2,
    "TurnID" : 2,
    "Reason" : "The guy in the 23 jersey is shooting three.",
    "Action" : {
        "Answer" : "23"
    }
}"""
        },
]

    # visualize_chat_prompt(history)

    original_question = error_case['question']
    first_turn = True
    turns = 0
    while True:
        if first_turn:
            batch_cur_caption = prompt_generation([error_case], torch.device('cuda:1'), caption_model, vis_processors, txt_processors, num_captions=30)
            first_turn = False
        else:
            batch_cur_caption = prompt_generation([error_case], torch.device('cuda:1'), caption_model, vis_processors, txt_processors, num_captions=15)
        torch.cuda.empty_cache()
        # get first caption
        cur_caption = batch_cur_caption[0]

        history.append({
            'role': 'user',
            'content': f"""{{
    "QueryID" : 3,
    "Question" : "{original_question}" 
    "Caption" : {json.dumps(cur_caption.split("."))},
}}"""
        })
        

        with torch.no_grad():
            batch_history, batch_cur_responese = answer_generation([history])
        # get first history and response
        history = batch_history[0]
        cur_responese = batch_cur_responese[0]

        # print(cur_responese)
        # TODO: change the 'question' field in error_case according to cur_response

        visualize_chat_prompt(history[-2:])
        try:
            cur_responese_dict = json.loads(cur_responese)
            if 'Answer' in cur_responese_dict['Action']:
                pred = cur_responese_dict['Action']['Answer']
                return pred, history, "answer"
            else:
                assert 'Search' in cur_responese_dict['Action']
                error_case['question'] = cur_responese_dict['Action']['Search']
                turns += 1
                # Tricky
                if turns > 5:
                    return error_case['question'], history, "question"

        except Exception:
            try:
                print("----Error in parsing response----")
                print(cur_responese)
                answer = re.search(r'(?:\'|\")Answer(?:\'|\")(.*)', cur_responese)
                if answer is not None:
                    pred = answer.group(1).strip().strip(':').strip('"')
                    return pred, history, "answer"
                question = re.search(r'(?:\'|\")Search(?:\'|\")(.*)', cur_responese)
                assert question is not None
                error_case['question'] = question.group(1).strip().strip(':').strip('"')
            except Exception:
                return cur_responese, history, "full response"


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='aokvqa')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--bsz', type=int, default=8)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--stage', type=str)

    args = parser.parse_args()
    # main(dataset_name=args.dataset_name, split=args.split, bsz=args.bsz, device=args.device, stage=args.stage)

    # with open('error_cases.json', 'r') as f:
    #     error_cases = json.load(f)
    model_name = f"llama2/Llama-2-13b-chat-hf"

    with open('hbz.json', 'r') as f:
        dataset = json.load(f)
    result = soft_acc(dataset, dataset)
    print(sum(result) / len(result))
    print(len(result))
    exit(0)

    ### Load Img2Prompt-VQA model
    caption_model, vis_processors, txt_processors = load_model_and_preprocess(name="img2prompt_vqa", model_type="base", is_eval=True, device=torch.device('cuda:1'))

    ### load valid dataset
    dataset_path = "datasets/" + args.dataset_name
    split = args.split
    dataset = load_aokvqa(dataset_path, split, version='v1p0', indices=None)

    # index = random.sample(range(len(dataset)),100)
    # extract error cases
    # with open('result_5_10.txt', 'r') as f:
    #     scores = json.loads(f.readline())
    # assert len(scores) == len(dataset)
    # index = [i for i in range(len(dataset)) if scores[i] != 0]

    # val_dataset = [dataset[i] for i in index]
    val_dataset = dataset
    # val_dataset = val_dataset[:100]

    answer = []

    # use tqdm to visualize progress
    try:
        with tqdm(total = len(val_dataset)) as pbar:
            pbar.set_description('Processing:')
            for i, item in enumerate(val_dataset):
                if i < 0:
                    pbar.update(1)
                    continue
                single = {'direct_answers': item['direct_answers']}
                single['pred'], single['history'], single['pred_type'] = error_cases_analysis(deepcopy(item))
                answer.append(single)
                # print(single['pred'])
                # print(item['direct_answers'])
                with open('hbz.json', 'w') as f:
                    json.dump(answer, f, indent=4)
                pbar.update(1)

    except Exception as e:
        print(e)
        result = soft_acc(val_dataset, answer)
        print(sum(result) / len(val_dataset))
        with open('hbz.json', 'w') as f:
            json.dump(answer, f, indent=4)

    result = soft_acc(val_dataset, answer)
    print(sum(result) / len(val_dataset))
    with open('hbz.json', 'w') as f:
        json.dump(answer, f, indent=4)

    
