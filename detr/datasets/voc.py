"""Pascal VOC 2007 and 2012 dataset."""
import json
from copy import deepcopy
from io import BytesIO
from pathlib import Path

import torch
import torchvision
from detr.datasets import coco
from PIL import Image
from pycocotools.coco import COCO



class VOCDetection(torchvision.datasets.VOCDetection):

    def __init__(self, root, year, image_set, transforms, in_memory=True):
        super().__init__(root, year, image_set)
        self.in_memory = in_memory
        self._transforms = transforms
        coco_file = (root / "VOCdevkit" / "external_PASCAL_VOC_coco_format" / 
                     "PASCAL_VOC" / f"pascal_{image_set}{year}.json")
        self.coco = COCO(coco_file)
        self.name_to_classid = {
            'aeroplane': 1,    'cat': 8,           'person': 15,
            'bicycle': 2,      'chair': 9,         'pottedplant': 16,
            'bird': 3,         'cow': 10,          'sheep': 17,
            'boat': 4,         'diningtable': 11,  'sofa': 18,
            'bottle': 5,       'dog': 12,          'train': 19,
            'bus': 6,          'horse': 13,        'tvmonitor': 20,
            'car': 7,          'motorbike': 14,
        }
        with open(coco_file, 'r') as f:
            coco_labels = json.load(f)
            self.filename_to_id = {l["file_name"]: l["id"] for l in coco_labels['images']}
        self.cache = [None] * len(self.filename_to_id)

    def __getitem__(self, idx):
        if not self.in_memory:
            img, target = super().__getitem__(idx)
        else:
            cached = self.cache[idx]
            if cached is None:
                img, target = super().__getitem__(idx)
                buffer = BytesIO()
                img.save(buffer, "JPEG")
                self.cache[idx] = (buffer, deepcopy(target))
            else:
                img, target = cached
                img = Image.open(img)

        img, target = self._prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target  

    def _prepare(self, image, target):
        w, h = image.size
        image_id = self.filename_to_id[target["annotation"]["filename"]]
        image_id = torch.tensor([image_id])

        annotation = target["annotation"]["object"]
        boxes = [obj["bndbox"] for obj in annotation]
        boxes = [[int(b["xmin"]), int(b["ymin"]), int(b["xmax"]), int(b["ymax"])] for b in boxes]
        # Boxes are in topleft bottom right format.
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [self.name_to_classid[obj["name"]] for obj in annotation]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in annotation])
        target["iscrowd"] = iscrowd[keep]
        # Skip area for now, should only be needed in eval.
        #area = torch.tensor([obj["area"] for obj in annotation])
        #target["area"] = area[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return image, target


def build(image_set, args):
    root = Path(args.data_path)
    # TODO: trainval and test?
    assert root.exists(), f'provided VOC path {root} does not exist'
    transforms = transforms=coco.make_coco_transforms(image_set)
    if image_set == "train":
        dataset =  torch.utils.data.ConcatDataset([
            VOCDetection(root, '2012', image_set='train', transforms=transforms),
            VOCDetection(root, '2012', image_set='val', transforms=transforms),
            VOCDetection(root, '2007', image_set='train', transforms=transforms),
            VOCDetection(root, '2007', image_set='val', transforms=transforms)])
    elif image_set == "val":
        dataset = VOCDetection(root, '2007', image_set='test', transforms=transforms)
    return dataset