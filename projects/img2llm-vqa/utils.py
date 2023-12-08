import os
import json
from PIL import Image
from networkx import lexicographic_product
import re

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
        num_match = sum([pred_answer == ans for ans in d['direct_answers']])
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