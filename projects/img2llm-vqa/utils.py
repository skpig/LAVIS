import os
import json
from PIL import Image
from networkx import lexicographic_product
import re
import openai

openai.api_key = "EMPTY"
# openai.base_url = "http://localhost:8000/v1"
openai.api_base = "http://localhost:8000/v1"

def chat_completion(history, max_tokens=200, temperature=0.):
    try:
        completion = openai.ChatCompletion.create(
            model='llama2_chat_13b',
            messages=history,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95,
            n=1
        )
    
        # print the completion
        return completion.choices[0].message.content
    except Exception as e:
        return "[Answer] Error, iteration too long."

def visualize_chat_prompt(history):
    """
    Format of history:
    [
            {"role": "system", "content": "Be a good boy."}
            {"role": "user", "content": "I am going to Paris, what should I see?"},
            {"role": "assistant", "content": "blah blah blah"},
            {"role": "user", "content": "What is so great about #1?"},
    ]
    """
    for i, turn in enumerate(history):
        if turn['role'] == 'system':
            print(f"===System===\n{turn['content']}")
        elif turn['role'] == 'user':
            print(f"===User===\n{turn['content']}")
        elif turn['role'] == 'assistant':
            print(f"===Assistant===\n{turn['content']}")
        else:
            raise NotImplementedError


def _chat_prompt(dialog, tokenizer):
    """
    Format of one history:
    [
            {"role": "system", "content": "Be a good boy."}
            {"role": "user", "content": "I am going to Paris, what should I see?"},
            {"role": "assistant", "content": "blah blah blah"},
            {"role": "user", "content": "What is so great about #1?"},
    ]
    """
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
    UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

    if dialog[0]["role"] == "system":
        dialog = [
            {
                "role": dialog[1]["role"],
                "content": B_SYS
                + dialog[0]["content"]
                + E_SYS
                + dialog[1]["content"],
            }
        ] + dialog[2:]
    assert all([msg["role"] == "user" for msg in dialog[::2]]) and \
           all([msg["role"] == "assistant" for msg in dialog[1::2]])
    
    history = ''.join([
        tokenizer.bos_token + f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} " + tokenizer.eos_token
        for prompt, answer in zip(
            dialog[::2],
            dialog[1::2],
        )
    ])

    assert (dialog[-1]["role"] == "user"), f"Last message must be from user, got {dialog[-1]['role']}"
    history += tokenizer.bos_token + f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"

    return history

def batch_chat_prompt(batch_history, tokenizer):
    return [_chat_prompt(history, tokenizer) for history in batch_history]

def postprocess_Answer(text):
    try: 
        answer = re.search(r'(.*?)(\.|\n)', text).group(1).strip()
        return answer
    except:
        return ''


def soft_acc(dataset, samples):
    acc = []
    for d, s in zip(dataset, samples):
        if 'direct_answers' not in d:
            continue
        pred_answer = postprocess_Answer(s['pred'])
        num_match = sum([pred_answer.find(ans) != -1 for ans in d['direct_answers']])
        acc.append(min(1.0, num_match/3.0))
        
    return acc
     

def load_aokvqa(aokvqa_dir, split, version='v1p0', indices=None):
    assert split in ['train', 'val', 'test', 'test_w_ans']
    dataset = json.load(open(
        os.path.join(aokvqa_dir, f"aokvqa_{version}_{split}.json")
    ))

    print(len(dataset)) 

    rtn = []
    for i, d in enumerate(dataset):
        if indices is not None and i not in indices:
            continue
        image_path = get_coco_path(split, d['image_id'])
        d['image_path'] = image_path
        if 'choices' in d and 'correct_choice_idx' in d:
            d['answer'] = d['choices'][d['correct_choice_idx']]
        else:
            d['answer'] = None
        # Corrrect: cab

        rtn.append(d)

    return rtn

def get_coco_path(split, image_id, coco_dir='/home/huangbz/LAVIS/projects/img2llm-vqa/datasets/coco'):
    return os.path.join(coco_dir, f"{split}2017", f"{image_id:012}.jpg")


def load_vqav2():
    pass

if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    load_aokvqa('./datasets/aokvqa', 'val')
    load_aokvqa('./datasets/aokvqa', 'test')