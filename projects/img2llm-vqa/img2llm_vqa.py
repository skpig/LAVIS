import sys
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

from utils import load_aokvqa, soft_acc

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
    gradcam = samples['gradcams'].reshape(24,24).numpy() # [576] -> [24, 24]， 获得最后一层feature的gradcam

    avg_gradcam = getAttMap(norm_img, gradcam, blur=True)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(avg_gradcam)
    ax.set_yticks([])
    ax.set_xticks([])
    print('Question: {}'.format(questions))

    fig.savefig('tmp.png')

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
    samples = {"image": images, "text_input": questions, "golden_answer": [item['answer'] for item in batch_data]}

    # #### 1. Image-Question Matching 
    # Compute the relevancy score of image patches with respect to the question using GradCAM
    # 计算GradCAM
    samples = model.forward_itm(samples=samples) # a dict of {image: [bsz, 3, H, W], text_input: a list of string, gradcams: [bsz, 576=24*24]}

    # #### 2. Image Captioning
    # Generate question-guided captions based on the relevancy score
    samples = model.forward_cap(samples=samples, num_captions=100, num_patches=20) # add a field 'captions': a list of bsz lists of strings
    # print('Examples of question-guided captions: ')
    # print(samples['captions'][0][:5])


    # #### 3. Question Generation
    # Generate synthetic questions using the captions
    with torch.no_grad():
        samples = model.forward_qa_generation(samples)
    # print('Sample Question: {} \nSample Answer: {}'.format(samples['questions'][:5], samples['answers'][:5]))


    # #### 4. Prompt Construction
    # Prepare the prompts for LLM

    # ### Generate answer by calling `predict_answers()` directly
    # add both contexts, (question, answer) exemplars as prompt
    samples = model.prompts_construction(samples) 

    del samples['image']
    samples['gradcams'] = samples['gradcams'].cpu()
    # release gradients
    model.zero_grad(set_to_none=True)

    rtn = [{k: v[i] for k, v in samples.items()} for i in range(len(batch_data))]

    return rtn

def answer_generation(model, tokenizer, samples, device):

    Img2Prompts = [item['Img2Prompts'] for item in samples]

    Img2Prompt_input = tokenizer(Img2Prompts, padding='longest', truncation=True, return_tensors="pt").to(device)

    assert (len(Img2Prompt_input.input_ids[0])+20) <=2048
    # print(len(question_input.attention_mask[0]))

    outputs_list  = []
    outputs = model.generate(input_ids=Img2Prompt_input.input_ids,
                            attention_mask=Img2Prompt_input.attention_mask,
                            max_length=20+len(Img2Prompt_input.input_ids[0]),
                            return_dict_in_generate=True,
                            output_scores = True
                            )
    outputs_list.append(outputs)

    preds = tokenizer.batch_decode(outputs.sequences[:, len(Img2Prompt_input.input_ids[0]):])

    #pred_answers, caption, gradcam = model.predict_answers(samples, num_captions=50, num_patches=20)
    #print('Question: {} \nPredicted answer: {}'.format(question, pred_answers[0]))

    for i, item in enumerate(samples):
        item['pred'] = preds[i]
    
    return samples

def prompt_generation_worker(dataset_name, split, worker_id, split_dataset, bsz):
    device = torch.device(f"cuda:{worker_id}")
    def batch_data(dataset, bsz):
        for i in range(0, len(dataset), bsz):
            yield dataset[i:i+bsz]
    
    if os.path.exists(os.path.join(OUTPUT_DIR, f"{dataset_name}_{split}.pt")):
        rtn =  torch.load(os.path.join(OUTPUT_DIR, f"{dataset_name}_{split}.pt"))
    else:
        rtn = []
    finished_indices = len(rtn) // bsz
    
    # ### Load Img2Prompt-VQA model
    model, vis_processors, txt_processors = load_model_and_preprocess(name="img2prompt_vqa", model_type="base", is_eval=True, device=device)

    # # Load images
    # for d in split_dataset:
    #     d['image'] = Image.open(d['image_path'])
    for i, batch in enumerate(batch_data(split_dataset, bsz)):
        if i < finished_indices:
            continue
        rtn.extend(prompt_generation(batch, device, model, vis_processors, txt_processors))
        # if worker_id == 0:
        #     print("Finished batch {}".format(i))
        print("Finished batch {}".format(i))
        torch.cuda.empty_cache()

        torch.save(rtn, os.path.join(OUTPUT_DIR, f"{dataset_name}_{split}.pt"))

    
    return rtn

def answer_generation_worker(model_name, dataset_name, split, worker_id, samples, bsz):
    device = torch.device(f"cuda:{worker_id}")
    def batch_data(dataset, bsz):
        for i in range(0, len(dataset), bsz):
            yield dataset[i:i+bsz]
    
    model,tokenizer = load_model(f'{MODEL_DIR}/{model_name}')
    model = model.to(device)

    model_name = model_name.split('/')[-1]

    output_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_{split}_{model_name}_answer.pt")
    if os.path.exists(output_path):
        rtn =  torch.load(output_path)
    else:
        rtn = []
    finished_indices = len(rtn) // bsz

    with torch.no_grad():
        for i, batch in enumerate(batch_data(samples, bsz)):
            if i < finished_indices:
                continue
            rtn.extend(answer_generation(model, tokenizer, batch, device))
            print("Finished batch {}".format(i))

            torch.save(rtn, output_path)

    return rtn    

def main(model_name='llama2/Llama-2-13b-chat-hf', dataset_name='aokvqa', split='val', bsz=2, device=0, stage=None, num_parallel_workers=1):

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
    main(dataset_name=args.dataset_name, split=args.split, bsz=args.bsz, device=args.device, stage=args.stage)