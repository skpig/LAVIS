"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 (CVPR 23') From Images to Textual Prompts: Zero-shot VQA with Frozen Large Language Models,
 by Jiaxian Guo, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Boyang Li, Dacheng Tao, Steven CH Hoi

 Initially referred as Img2prompt_vqa, later Img2LLM_vqa.
"""

import random

import spacy
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer

from lavis.common.dist_utils import download_cached_file
from lavis.common.registry import registry
from lavis.models.base_model import BaseModel
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam

open_pos = ["NOUN", "VERB", "ADJ", "ADV", "NUM"]



@registry.register_model("img2prompt_vqa")
class Img2PromptVQA(BaseModel):
    """
    Img2Prompt_VQA model consists of three submodels for zero-shot VQA:
        1. Image-questioning matching model
        2. Image captioning model
        3. Large Language model

    Supported model types:
        - base: BLIPITM, BLIPCaption, PNPUnifiedQAv2FiD (t5-base)
        - large: BLIPITM, BLIPCaption, PNPUnifiedQAv2FiD (t5-large)
        - 3b: BLIPITM, BLIPCaption, PNPUnifiedQAv2FiD (t5-3b)

    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("img2prompt_vqa", "base", is_eval=True)
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/img2prompt-vqa/img2prompt_vqa_base.yaml",
    }

    def __init__(
        self,
        image_question_matching_model,
        image_captioning_model,
        question_generation_model,
        question_generation_tokenizer,
        offload_model=False,
    ):
        super().__init__()

        self.image_question_matching_model = image_question_matching_model
        self.image_captioning_model = image_captioning_model
        self.question_generation_model = question_generation_model
        self.question_generation_tokenizer = question_generation_tokenizer
        self.offload_model = offload_model
        self.nlp = spacy.load("en_core_web_sm")

    def forward_itm(self, samples, block_num=7):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - text_input (list): A list of strings of length batch_size
            block_num (int): The index of cross-attention block for gradcam computation.

        Returns:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - text_input (list): A list of strings of length batch_size
                - gradcams (torch.Tensor): A tensor of shape (batch_size, H*W)
        """
        image = samples["image"]
        question = [text.strip("?") for text in samples["text_input"]]
        tokenized_text = self.image_question_matching_model.tokenizer(
            question, padding="longest", truncation=True, return_tensors="pt"
        ).to(self.image_question_matching_model.device)
        with torch.set_grad_enabled(True):
            gradcams, _ = compute_gradcam(
                model=self.image_question_matching_model,
                visual_input=image,
                text_input=question,
                tokenized_text=tokenized_text,
                block_num=block_num,
            )

        gradcams = [gradcam_[1] for gradcam_ in gradcams]
        samples["gradcams"] = torch.stack(gradcams).reshape(
            samples["image"].size(0), -1
        ).clone().detach_()

        del gradcams
        del _

        return samples

    def itm_rank(self, image_embeds, image_atts, encoder_input_ids, match_head="itm"):
        # breakpoint()
        encoder_input_ids = encoder_input_ids.clone()
        encoder_input_ids = encoder_input_ids[:, self.prompt_length - 1 :]
        text_attention_mask = (encoder_input_ids != self.tokenizer.pad_token_id).long()

        if match_head == "itm":
            # encoder_input_ids = encoder_input_ids.clone()
            with torch.no_grad():
                encoder_input_ids[:, 0] = self.tokenizer.enc_token_id
                output = self.text_encoder(
                    encoder_input_ids,
                    attention_mask=text_attention_mask,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
                itm_output = self.itm_head(output.last_hidden_state[:, 0, :])
                return itm_output.clone().detach_()  # , mask, token_length

        elif match_head == "itc":
            with torch.no_grad():
                encoder_input_ids[:, 0] = self.tokenizer.cls_token_id
                text_output = self.text_encoder(
                    encoder_input_ids,
                    attention_mask=text_attention_mask,
                    return_dict=True,
                    mode="text",
                )
                image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
                text_feat = F.normalize(
                    self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
                )

                sim = image_feat @ text_feat.t()
                return sim.clone().detach_()

    def forward_cap(
        self,
        samples,
        cap_max_length=20,
        cap_min_length=0,
        top_p=1,
        top_k=50,
        repetition_penalty=1.0,
        num_captions=100,
        num_patches=20,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - text_input (list): A list of strings of length batch_size
                - gradcams (torch.Tensor): A tensor of shape (batch_size, H*W)
            cap_max_length (int): The maximum length of the caption to be generated.
            cap_min_length (int): The minimum length of the caption to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            top_k (float): The number of the highest probability tokens for top-k sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions generated for each image.
            num_patches (int): Number of patches sampled for each image.

        Returns:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - text_input (list): A list of strings of length batch_size
                - gradcams (torch.Tensor): A tensor of shape (batch_size, H*W)
                - captions (nested list): A nested list of strings of total length batch_size * num_captions
        """
        with torch.no_grad():
            encoder_out = self.image_captioning_model.forward_encoder(samples) # [bsz, num_patch=1+24*24=577, dim=1024]
        captions = [[] for _ in range(encoder_out.size(0))] # bsz lists

        min_num_captions = 0

        while min_num_captions < num_captions:
            encoder_out_samples = []
            # For each caption
            for i in range(num_captions):
                # sample patches according to gradcam
                patch_id = (
                    torch.multinomial(
                        samples["gradcams"].to(self.image_captioning_model.device),
                        num_patches,
                    ).reshape(encoder_out.size(0), -1)
                    + 1
                )
                patch_id = (
                    patch_id.sort(dim=1)
                    .values.unsqueeze(-1)
                    .expand(-1, -1, encoder_out.size(2))
                )
                encoder_out_sample = torch.gather(encoder_out, 1, patch_id) # [bsz, num_patch, dim=1024]
                encoder_out_samples.append(encoder_out_sample) 

            # create encoder-hidden-states for multimodal-cross-attention
            stacked = torch.stack(encoder_out_samples, dim=1) # [bsz, num_caption, num_patch, dim=1024]
            image_embeds = torch.flatten(
                stacked, start_dim=0, end_dim=1
            )  # (bsz*num_caption, num_patch, dim)
            image_attn_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                self.image_captioning_model.device
            )
            model_kwargs = {
                "encoder_hidden_states": image_embeds,
                "encoder_attention_mask": image_attn_mask,
            }

            # create input_ids for text decoder 
            prompt = [self.image_captioning_model.prompt] * image_embeds.size(0)
            prompt = self.image_captioning_model.tokenizer(
                prompt, return_tensors="pt"
            ).to(self.image_captioning_model.device)
            prompt.input_ids[:, 0] = self.image_captioning_model.tokenizer.bos_token_id
            prompt.input_ids = prompt.input_ids[:, :-1]

            # generate (bsz * num_caption) captions according to (num_patches) image patches
            decoder_out = self.image_captioning_model.text_decoder.generate(
                input_ids=prompt.input_ids,
                max_length=cap_max_length,
                min_length=cap_min_length,
                do_sample=True,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=1,
                eos_token_id=self.image_captioning_model.tokenizer.sep_token_id,
                pad_token_id=self.image_captioning_model.tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                **model_kwargs
            )

            # filter caption according to itm score (more than 0.5)
            itm_outputs = self.image_question_matching_model.itm_rank(
                image_embeds, image_attn_mask, encoder_input_ids=decoder_out
            )  # [bsz * num_caption]

            outputs = self.image_captioning_model.tokenizer.batch_decode(
                decoder_out, skip_special_tokens=True
            )

            for counter, output in enumerate(outputs):
                ind = counter // num_captions
                if len(captions[ind]) < num_captions:
                    caption = output[len(self.image_captioning_model.prompt) :]
                    # no duplicate captions
                    overlap_caption = [1 for caps in captions[ind] if caption in caps]
                    if (
                        len(overlap_caption) == 0 and itm_outputs[counter] >= 0.5
                    ):  # image filter
                        captions[ind].append(caption)

            min_num_captions = min([len(i) for i in captions])

        samples["captions"] = captions

        return samples

    def answer_extraction(self, captions, num_question_generation=30):
        cap_use = "" # ??? why concat all captions?
        ans_to_cap_dict = {} # record the caption id for each answer
        answers = []
        for cap_idx, cap in enumerate(captions):
            # print(cap)
            cap_use += cap
            cap = cap.strip().strip(".")
            # print(cap)
            cap = self.nlp(cap) # use spacy for parsing
            # extract each token
            for token in cap:  # Noun /Verb/Adj//NUM
                if token.pos_ in open_pos:
                    if token.text.lower() not in ans_to_cap_dict:
                        ans_to_cap_dict[token.text.lower()] = [cap_idx]
                    else:
                        if cap_idx not in ans_to_cap_dict[token.text.lower()]:
                            ans_to_cap_dict[token.text.lower()].append(cap_idx)
                    answers.append(token.text)
            # extract entity as answer
            for ent in cap.ents:
                if ent.text not in answers:
                    if ent.text.lower() not in ans_to_cap_dict:
                        ans_to_cap_dict[ent.text.lower()] = [cap_idx]
                    else:
                        if cap_idx not in ans_to_cap_dict[ent.text.lower()]:
                            ans_to_cap_dict[ent.text.lower()].append(cap_idx)
                    answers.append(ent.text)
            # extract chunk
            for chunk in cap.noun_chunks:
                if len(chunk.text.split()) < 4:
                    if chunk.text.lower() not in ans_to_cap_dict:
                        ans_to_cap_dict[chunk.text.lower()] = [cap_idx]
                    else:
                        if cap_idx not in ans_to_cap_dict[chunk.text.lower()]:
                            ans_to_cap_dict[chunk.text.lower()].append(cap_idx)
                    #                 print(chunk.text)
                    answers.append(chunk.text)
        answers = sorted(answers, key=answers.count, reverse=True)
        real_answers = []
        for i in answers:
            i = i + "."
            if i not in real_answers:
                real_answers.append(i)

        contexts_for_question_generation = []
        answers = []
        for ans in real_answers[
            :num_question_generation
        ]:  # Generate questions for 30 answers with max frequencies.
            contexts_for_question_generation.append(
                "answer: %s  context: %s." % (ans, cap_use)
            )
            answers.append(ans)
        contexts_for_question_generation.append(
            "answer: %s  context: %s." % ("yes.", cap_use)
        )
        answers.append("yes.")
        return contexts_for_question_generation, answers, ans_to_cap_dict

    def forward_qa_generation(self, samples):
        samples["questions"] = []
        samples["answers"] = []
        samples["ans_to_cap_dict"] = []
        for captions in samples["captions"]:
            (
                contexts_for_question_generation, # answer: a1 ... context: cap1, cap2, ... (Noted that all captions are in the context)
                answers, # a1, a2, ...
                ans_to_cap_dict, # a1: [cap1, cap2, ...]
            ) = self.answer_extraction(captions) 
            inputs = self.question_generation_tokenizer(
                contexts_for_question_generation,
                padding="longest",
                truncation=True,
                max_length=1536,
                return_tensors="pt",
            ).to(self.device)
            question_size = inputs.input_ids.shape[0]
            cur_b = 0
            true_input_size = 8
            outputs_list = []
            while cur_b < question_size:
                # generate in a batch of `true_input_size`
                outputs = self.question_generation_model.generate(
                    input_ids=inputs.input_ids[cur_b : cur_b + true_input_size],
                    attention_mask=inputs.attention_mask[cur_b : cur_b + true_input_size],
                    num_beams=3,
                    max_length=30,
                )
                questions = self.question_generation_tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )
                outputs_list += questions
                cur_b += true_input_size
            questions = outputs_list
            samples["questions"].append(questions)
            samples["answers"].append(answers)
            samples["ans_to_cap_dict"].append(ans_to_cap_dict)

            del inputs
            del outputs
        # results.append({"question_id": ques_id, "question":questions,"answer":answers})
        return samples
    # TODO: Maybe buggy
    def create_context_prompt(self, samples, num_caps_per_img=30):
        samples["Context_Prompt"] = []
        for i in range(len(samples["answers"])):
            ans_dict_queid = samples["ans_to_cap_dict"][i]
            # print(ans_dict_queid)
            caption = samples["captions"][i]
            answers = samples["answers"][i]
            Context_Prompt = ""
            mycontexts_id = []
            for idx in range(num_caps_per_img):
                cap_id_list = ans_dict_queid.get(
                    answers[(len(answers) - 1 - idx) % len(answers)][:-1].lower(), [0] # use [:-1] to remove '.'
                )
                for cap_id in cap_id_list:
                    if cap_id not in mycontexts_id:
                        Context_Prompt += caption[cap_id]
                        mycontexts_id.append(cap_id)
                        break  # We just take one cap for each answer
            samples["Context_Prompt"].append(Context_Prompt)
        return samples["Context_Prompt"]

    def create_task_prompt(
        self, samples, question_type="neural", num_question_per_img=30
    ):
        samples["Task_Prompt"] = []
        for i in range(len(samples["answers"])):
            syn_question_queid = samples["questions"][i]
            syn_ans_queid = samples["answers"][i]
            Task_Prompt = ""
            for idx in range(num_question_per_img):
                # if config['random_question']:
                #     qa_idx = random.randint(0, len(syn_question_queid) - 1)
                # else:
                qa_idx = idx
                if (
                    question_type != "rule" and num_question_per_img > 0
                ):  ## yes and no questions for vqav2
                    # Task_Prompt += "Question:"
                    # Task_Prompt += syn_question_queid_next[-1]
                    # Task_Prompt += '\n'
                    # Task_Prompt += "Answer:no\n"
                    Task_Prompt += "Question:"
                    Task_Prompt += syn_question_queid[qa_idx]
                    Task_Prompt += "\n"
                    Task_Prompt += "Answer:"
                    Task_Prompt += syn_ans_queid[qa_idx]
                    Task_Prompt += "\n"
                if "question_type" == "rule":  # Rule-Based Question Generation
                    Noun_Questions = [
                        "What item is this in this picture?",
                        "What item is that in this picture?",
                    ]

                    Verb_Questions = [
                        "What action is being done in this picture?",
                        "Why is this item doing in this picture?",
                        "Which action is being taken in this picture?",
                        "What action is item doing in this picture?",
                        "What action is item performing in this picture?",
                    ]

                    Adj_Questions = [
                        "How to describe one item in this picture?",
                        "What is item's ADJ TYPE in this picture?",
                        "What is the ADJ TYPE in this picture?",
                    ]

                    Task_Prompt += "Question:"
                    doc = self.nlp(syn_ans_queid[(qa_idx) % len(syn_ans_queid)][:-1].lower())
                    if doc[-1].pos_ == "NOUN":
                        Task_Prompt += Noun_Questions[
                            random.randint(0, len(Noun_Questions) - 1)
                        ]
                    elif doc[-1].pos_ == "VERB":
                        Task_Prompt += Verb_Questions[
                            random.randint(0, len(Verb_Questions) - 1)
                        ]
                    elif doc[-1].pos_ == "ADJ":
                        Task_Prompt += Adj_Questions[
                            random.randint(0, len(Adj_Questions) - 1)
                        ]

                    Task_Prompt += "\n"

                    Task_Prompt += "Answer:"
                    Task_Prompt += syn_ans_queid[(qa_idx) % len(syn_ans_queid)][:-1].lower()
                    Task_Prompt += "\n"
            samples["Task_Prompt"].append(Task_Prompt)
        # print(Task_Prompt)
        return samples["Task_Prompt"]

    def prompts_construction(
        self,
        samples,
        question_type="neural",
        num_caps_per_img=30,
        num_question_per_img=30,
    ):
        Prompt = "Please reason the answer of the questions according to the given contexts.\n"

        Context_Prompts = self.create_context_prompt(samples, num_caps_per_img) # a batch of contexts

        Task_Prompts = self.create_task_prompt(
            samples, question_type, num_question_per_img
        ) # a batch of questions

        samples['Img2Prompts'] = [(
            Prompt
            + "Contexts:"
            + Context_Prompt
            + "\n"
            + Task_Prompt
            + "Question:"
            + input_question
            + "\nAnswer:"
        ) for Context_Prompt, Task_Prompt, input_question in zip(Context_Prompts, Task_Prompts, samples['text_input'])]

        return samples

    def prepare_LLM_input(
        self,
        samples,
        num_beams=1,
        inference_method="generate",
        max_len=20,
        min_len=0,
        internal_bsz_fid=1,
        num_captions=50,
        num_captions_fid=1,
        cap_max_length=20,
        cap_min_length=10,
        top_k=50,
        top_p=1,
        repetition_penalty=1,
        num_patches=20,
        block_num=7,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W). Default H=480, W=480.
                - text_input (str or [str]): String or a list of strings, each string is a question.
                                             The number of questions must be equal to the batch size. If a single string, will be converted to a list of string, with length 1 first.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            inference_method (str): Inference method. Must be "generate". The model will generate answers.
            max_len (int): Maximum length of generated answers.
            min_len (int): Minimum length of generated answers.
            internal_bsz_fid (int): Internal batch size when using FiD decoding.
            num_captions (int): Number of captions generated for each image.
            num_captions_fid (int): Number of captions concatenated with a question during FiD decoding.
            cap_max_length (int): The maximum length of the caption to be generated.
            cap_min_length (int): The minimum length of the caption to be generated.
            top_k (float): The number of the highest probability tokens for top-k sampling.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_patches (int): Number of patches sampled for each image.
            block_num (int): The index of cross-attention block for gradcam computation.

        Returns:
            List: A list of strings, each string is an answer.
            gradcams (torch.Tensor): A tensor of shape (batch_size, H*W)
            captions (nested list): A nested list of strings of total length batch_size * num_captions
        """
        assert inference_method in [
            "generate",
        ], "Inference method must be 'generate', got {}.".format(inference_method)

        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        assert len(samples["text_input"]) == samples["image"].size(
            0
        ), "The number of questions must be equal to the batch size."

        samples = self.forward_itm(samples, block_num=block_num)

        samples = self.forward_cap(
            samples,
            cap_max_length=cap_max_length,
            cap_min_length=cap_min_length,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_captions=num_captions,
            num_patches=num_patches,
        )

        if self.offload_model:
            samples["image"] = samples["image"].to("cpu")
            self.image_question_matching_model.to("cpu")
            self.image_captioning_model.to("cpu")
        torch.cuda.empty_cache()

        pred_answers = self.forward_qa(
            samples,
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            internal_bsz_fid=internal_bsz_fid,
            num_captions=num_captions,
            num_captions_fid=num_captions_fid,
        )

        if self.offload_model:
            self.image_question_matching_model.to(self.question_answering_model.device)
            self.image_captioning_model.to(self.question_answering_model.device)

        return pred_answers, samples["captions"], samples["gradcams"]

    @classmethod
    def from_config(cls, model_config):
        itm_config = model_config.image_question_matching_model
        cap_config = model_config.image_captioning_model

        itm_cls = registry.get_model_class(itm_config.arch)
        cap_cls = registry.get_model_class(cap_config.arch)

        image_question_matching_model = itm_cls.from_config(itm_config)
        image_captioning_model = cap_cls.from_config(cap_config)

        question_generation_tokenizer = T5Tokenizer.from_pretrained(
            "google/t5-large-lm-adapt"
        )
        question_generation_model = T5ForConditionalGeneration.from_pretrained(
            "google/t5-large-lm-adapt"
        )
        cached_file = download_cached_file(
            "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/projects/img2prompt/T5_large_QG.pth",
            check_hash=False,
            progress=True,
        )
        checkpoint = torch.load(cached_file, map_location="cpu")
        state_dict = checkpoint["model"]
        question_generation_model.load_state_dict(state_dict)
        model = cls(
            image_question_matching_model=image_question_matching_model,
            image_captioning_model=image_captioning_model,
            question_generation_model=question_generation_model,
            question_generation_tokenizer=question_generation_tokenizer,
            offload_model=False,
        )

        return model
