
vqa_data_dir = "path/to/vqav2/train2014"
vqa_questions = "path/to/vqav2/v2_OpenEnded_mscoco_train2014_questions.json"
vqa_ann = "path/to/vqav2/v2_mscoco_train2014_annotations.json"
coco_data_dir = "path/to/coco/train2014"
coco_ann = "path/to/coco/annotations/captions_train2014.json"

import json
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class CaptionDataset(Dataset):
    def __init__(
        self,
        image_train_dir_path,
        annotations_path,
        tokenizer=None,
        transforms=None,
        seed=123,
        is_train=True,
        dataset_name='coco',
        image_val_dir_path=None,
    ):
        self.image_train_dir_path = image_train_dir_path
        self.image_val_dir_path = image_val_dir_path
        self.annotations = []
        self.is_train = is_train
        self.dataset_name = dataset_name
        self.seed = seed
        random.seed(self.seed)
        full_annotations = json.load(open(annotations_path))
        self.tokenizer = tokenizer
        self.transforms = transforms
        print(len(full_annotations["images"]), len(full_annotations["annotations"]))
        self.id2path = {}
        self.id2caption = {}
        for i in range(len(full_annotations["images"])):
            self.id2path[full_annotations["images"][i]["id"]] = os.path.join(
                self.image_train_dir_path, full_annotations["images"][i]["file_name"])
        self.image_ids = list(self.id2path.keys())
        for i in range(len(full_annotations["annotations"])):
            image_id = full_annotations["annotations"][i]["image_id"]
            if image_id not in self.id2caption:
                self.id2caption[image_id] = [full_annotations["annotations"][i]['caption']]
            else:
                self.id2caption[image_id].append(full_annotations["annotations"][i]['caption'])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image = Image.open(self.id2path[self.image_ids[idx]])
        image.load()
        caption = random.choice(self.id2caption[self.image_ids[idx]])
        return {
            "image": image,
            "caption": caption,
            "image_id": self.image_ids[idx]
        }
    
    def get_caption_prompt(self, caption=None):
        return f"A photo of {caption if caption is not None else ''}"
    
    def collator(self, samples):
        images = torch.stack([self.transforms(s['image']) for s in samples], dim=0)
        text = [self.get_caption_prompt(s['caption']) for s in samples]
        text_tensors, attention_mask = self.tokenizer(text)
        return images, (text_tensors, attention_mask)


class VQADataset(Dataset):
    def __init__(
        self, image_dir_path, question_path, annotations_path, tokenizer=None, transforms=None, seed=123, is_train=True, dataset_name='vqav2'
    ):
        self.questions = json.load(open(question_path, "r"))["questions"]
        if annotations_path is not None:
            self.answers = json.load(open(annotations_path, "r"))["annotations"]
        else:
            self.answers = None
        self.image_dir_path = image_dir_path
        self.is_train = is_train
        self.dataset_name = dataset_name
        # self.img_coco_split = "train2014"
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.seed = seed
        random.seed(self.seed)
        if self.dataset_name in {"vqav2", "ok_vqa"}:
            self.img_coco_split = self.image_dir_path.strip("/").split("/")[-1]
            assert self.img_coco_split in {"train2014", "val2014", "test2015"}

    def __len__(self):
        return len(self.questions)

    def get_img_path(self, question):
        if self.dataset_name in {"vqav2", "ok_vqa"}:
            return os.path.join(
                self.image_dir_path,
                f"COCO_{self.img_coco_split}_{question['image_id']:012d}.jpg"
                if self.is_train
                else f"COCO_{self.img_coco_split}_{question['image_id']:012d}.jpg",
            )
        elif self.dataset_name == "vizwiz":
            return os.path.join(self.image_dir_path, question["image_id"])
        elif self.dataset_name == "textvqa":
            return os.path.join(self.image_dir_path, f"{question['image_id']}.jpg")
        else:
            raise Exception(f"Unknown VQA dataset {self.dataset_name}")

    def __getitem__(self, idx):
        question = self.questions[idx]
        img_path = self.get_img_path(question)
        image = Image.open(img_path)
        # image.load()
        results = {
            "image": image,
            "question": question["question"],
            "question_id": question["question_id"],
        }
        if self.answers is not None:
            answers = self.answers[idx]
            results["answers"] = [a["answer"] for a in answers["answers"]]
        return results
    
    def get_vqa_prompt(self, question, answer=None):
        return f"Question:{question} Short answer:{answer if answer is not None else ''}"
    
    def get_vqa_ques_prompt(self, question):
        return f"Question:{question} Short answer:"
    
    def collator(self, samples):
        images = torch.stack([self.transforms(s['image']) for s in samples], dim=0)
        text = [self.get_vqa_prompt(s['question'], random.choice(s['answers'])) for s in samples]
        text_tensors, attention_mask = self.tokenizer(text)
        B, T = attention_mask.shape
        ques = [self.get_vqa_ques_prompt(s['question']) for s in samples]
        _, ques_mask = self.tokenizer(ques)
        ques_len = ques_mask.sum(dim=1).unsqueeze(-1).expand(B, T)
        answer_mask = torch.ones_like(attention_mask)
        indices = torch.arange(answer_mask.shape[-1]).unsqueeze(0).expand(B, T)
        index_mask = indices < ques_len
        answer_mask.masked_fill_(index_mask, 0)
        answer_mask = answer_mask * attention_mask # both mask for attention and question
        return images, (text_tensors, attention_mask), answer_mask