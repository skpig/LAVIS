import sys
import torch
import os
import requests
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

from lavis.common.gradcam import getAttMap
from lavis.models import load_model_and_preprocess


def example_code():
    # ### Load an example image and question
    # img_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/projects/pnp-vqa/demo.png' 
    # raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')   
    raw_image = Image.open("./demo.png").convert("RGB")
    #display(raw_image.resize((400, 300)))
    question = "What item s are spinning which can be used to control electric?"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # ### Load Img2Prompt-VQA model
    model, vis_processors, txt_processors = load_model_and_preprocess(name="img2prompt_vqa", model_type="base", is_eval=True, device=device)

    # ### Preprocess image and text inputs
    # 预处理图片和文本
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device) # BLIP image processor处理后，图片大小 [1, 3, 384, 384]
    question = txt_processors["eval"](question)
    samples = {"image": image, "text_input": [question]}

    # #### 1. Image-Question Matching 
    # Compute the relevancy score of image patches with respect to the question using GradCAM
    # 计算GradCAM
    samples = model.forward_itm(samples=samples) # a dict of {image, text_input, gradcams: [1, 576]}
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
    print('Question: {}'.format(question))

    fig.savefig('tmp.png')

    # #### 2. Image Captioning
    # Generate question-guided captions based on the relevancy score
    samples = model.forward_cap(samples=samples, num_captions=50, num_patches=20)
    print('Examples of question-guided captions: ')
    print(samples['captions'][0][:5])


    # #### 3. Question Generation
    # Generate synthetic questions using the captions
    samples = model.forward_qa_generation(samples)
    print('Sample Question: {} \nSample Answer: {}'.format(samples['questions'][:5], samples['answers'][:5]))


    # #### 4. Prompt Construction
    # Prepare the prompts for LLM

    # ### Generate answer by calling `predict_answers()` directly
    # add both contexts, (question, answer) exemplars as prompt
    Img2Prompt = model.prompts_construction(samples)


    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
    def load_model(model_selection):
        model = AutoModelForCausalLM.from_pretrained(model_selection) #,
                                                    #  torch_dtype=torch.bfloat16) # OPTIMIZATION: Possible to use bfloat16
        tokenizer = AutoTokenizer.from_pretrained(model_selection, use_fast=False)
        return model,tokenizer
    def postprocess_Answer(text):
        for i, ans in enumerate(text):
            for j, w in enumerate(ans):
                if w == '.' or w == '\n':
                    ans = ans[:j].lower()
                    break
        return ans

    model,tokenizer = load_model('facebook/opt-2.7b')

    Img2Prompt_input = tokenizer(Img2Prompt, padding='longest', truncation=True, return_tensors="pt").to(device) # OPTIMIZATION: can load to another device

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




    pred_answer = tokenizer.batch_decode(outputs.sequences[:, len(Img2Prompt_input.input_ids[0]):])
    pred_answer = postprocess_Answer(pred_answer)
    print({"question": question, "answer": pred_answer})

    #pred_answers, caption, gradcam = model.predict_answers(samples, num_captions=50, num_patches=20)
    #print('Question: {} \nPredicted answer: {}'.format(question, pred_answers[0]))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    example_code()