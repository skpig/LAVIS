from copy import deepcopy
import sys
import re
import json
import torch
import os
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

def prompt_generation(batch_data, device, model, vis_processors, txt_processors):
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
    gradcam_visualization(image_lst[0], samples)

    # #### 2. Image Captioning
    # Generate question-guided captions based on the relevancy score
    samples = model.forward_cap(samples=samples, num_captions=10, num_patches=20) # add a field 'captions': a list of bsz lists of strings
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

    print("========================================")
    print("============Question====================")
    print("============Image: see image.png========")
    os.system(f"cp {error_case['image_path']} image.png")
    print("============Chat History================")

    # TODO: design the prompt history
    # see utils.prompt() for details
    history = [
        {
            'role': 'system',
            'content': """Each time I give you a query, including a [Question] and some [Captions] about an image. Try to answer the question by thinking step-by-step according to given contexts. 
1. You need to give the "[Reason]:reason" in each response. 
2. You can use the "[Search]:object" to ask for more image captions, where "object" is the term you want to look up in the image.
3. Output "[Answer]:answer" if you have found the answer. Otherwise, output "[Answer]:None"."""
        },
        {
            'role': 'user',
            'content': """[First Query]
[Captions] a fence of picket white boards with a gate. the house is fenced in in front of a white picketed fence. a white picket with pink roses in front of it.
[Question] What item is in front of the fence which can be used to fire fighting? """
        },
        {
            'role': 'assistant',
            'content': """[First Query]
[Reason] We know that there are pink roses in front of the fence, but we don't know the information about the yellow thing, we need the shape of the yellow object about it.
[Search] yellow object in front of the fence.
[Answer] None"""
        },
        {
            'role': 'user',
            'content': "[First Query]\n[Captions] a yellow fire hydrant in front of a white fence."
        },
        {
            'role': 'assistant',
            'content': """[First Query]
[Reason] The yellow fire hydrant is in front of the white fence, while it can also be used to put out fires.
[Search] None,
[Answer] Fire hydrant"""
        }
    ]

    visualize_chat_prompt(history)

    original_question = error_case['question']
    first_turn = True
    while True:
        batch_cur_caption = prompt_generation([error_case], torch.device('cuda:1'), caption_model, vis_processors, txt_processors)
        torch.cuda.empty_cache()
        # get first caption
        cur_caption = batch_cur_caption[0]

        # TODO: how to add cur_caption to history
        if first_turn:
            history.append({
                'role': 'user',
                'content': f"""[Second Query]
[Captions] {cur_caption}
[Question] {error_case['question']}"""
            })
            first_turn = False
        else:
            history.append({
                'role': 'user',
                'content': f"""[Second Query]
[Captions] {cur_caption}"""
            })
        

        with torch.no_grad():
            batch_history, batch_cur_responese = answer_generation([history])
        # get first history and response
        history = batch_history[0]
        cur_responese = batch_cur_responese[0]

        

        # TODO: change the 'question' field in error_case according to cur_response
        search_question = re.search(r'\[Search\] (.*)\n', cur_responese)
        if search_question is not None:
            error_case['question'] = search_question.group(1)
        else:
            error_case['question'] = original_question

        visualize_chat_prompt(history[-2:])

        # input("Press Enter to continue...")
        group = re.search(r'\[Answer\] (.*)', cur_responese)
        if group is not None and group.group(1).find('None') == -1:
            return cur_responese, history



if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='aokvqa')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--bsz', type=int, default=8)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--stage', type=str)

    args = parser.parse_args()
    # main(dataset_name=args.dataset_name, split=args.split, bsz=args.bsz, device=args.device, stage=args.stage)

    with open('error_cases.json', 'r') as f:
        error_cases = json.load(f)
    model_name = f"llama2/Llama-2-13b-chat-hf"

    ### Load Img2Prompt-VQA model
    caption_model, vis_processors, txt_processors = load_model_and_preprocess(name="img2prompt_vqa", model_type="base", is_eval=True, device=torch.device('cuda:1'))


    for error_case in error_cases:
        error_case['new_pred'], error_case['history'] = error_cases_analysis(deepcopy(error_case))
    with open('error_cases_ours.json', 'w') as f:
        json.dump(error_cases, f, indent=4)

    
