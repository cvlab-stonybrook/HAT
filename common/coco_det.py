import os
import torch
import torch.utils.data
import torchvision
import numpy as np
from PIL import Image
from .utils import get_IoM

class COCODetHelper(object):
    def __init__(self, coco_annos):
        self.coco_annos = coco_annos
        self.num_classes = 80
        self.class_name = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.valid_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
            82, 84, 85, 86, 87, 88, 89, 90]
        self.cat_ids = {v: i for i, v in enumerate(self.valid_ids)}
        self.cat_name_ids = {v: i for i, v in enumerate(self.class_name)}
        
    def get_annos_by_img_name(self, img_name, padded_img_height, padded_img_width):

        def update_bbox(bbox, padded_img_height, padded_img_width, h, w):
            target_ratio = padded_img_width / float(padded_img_height)
            ratio = w / float(h)
            delta_w, delta_h = 0, 0

            if ratio > target_ratio:
                new_w = padded_img_width
                new_h = int(new_w / ratio)
                delta_h = padded_img_height - new_h
            else:
                new_h = padded_img_height
                new_w = int(new_h * ratio)
                delta_w = padded_img_width - new_w

            # padding with 0
            top = delta_h//2
            left = delta_w//2

            # compute label as shape changes
            bbox = [x * new_w / float(w) for x in bbox]
            bbox[0], bbox[1] = bbox[0] + left, bbox[1] + top
            return bbox

        img_id = int(img_name[:-4])

        # List: get annotation id from coco
        coco_annos_train, coco_annos_val = self.coco_annos
        ann_ids = coco_annos_train.getAnnIds(imgIds=img_id)
        coco = coco_annos_train
        if len(ann_ids) == 0:
            ann_ids = coco_annos_val.getAnnIds(imgIds=img_id)
            coco = coco_annos_val

        img_info = coco.loadImgs(img_id)[0]

        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            bbox = update_bbox(coco_annotation[i]['bbox'], 
                               padded_img_height, padded_img_width,
                               img_info['height'], img_info['width'])
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = xmin + bbox[2]
            ymax = ymin + bbox[3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = np.array(boxes, dtype=np.float32)

        # clip bbox to image plane
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, padded_img_width-1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, padded_img_height-1)

        labels = np.array([anno['category_id'] for anno in coco_annotation])

        # Tensorise img_id
        img_id = np.array([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = np.array(areas, dtype=np.float32)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        return my_annotation

    def gaussian_radius(self, det_size, min_overlap=0.7):
        height, width = det_size

        a1  = 1
        b1  = (height + width)
        c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1  = (b1 + sq1) / 2

        a2  = 4
        b2  = 2 * (height + width)
        c2  = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2  = (b2 + sq2) / 2

        a3  = 4 * min_overlap
        b3  = -2 * min_overlap * (height + width)
        c3  = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3  = (b3 + sq3) / 2
        return min(r1, r2, r3)

    def draw_umich_gaussian(self, heatmap, center, radius, k=1):
        def gaussian2D(shape, sigma=1):
            m, n = [(ss - 1.) / 2. for ss in shape]
            y, x = np.ogrid[-m:m+1,-n:n+1]

            h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            return h

        diameter = 2 * radius + 1
        gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

        return heatmap

    def create_centermap_target(self, img_name, output_h, output_w):
        annos = self.get_annos_by_img_name(img_name, output_h, output_w)
        hm = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
        num_objs = len(annos['boxes'])

        gt_det = []
        for k in range(num_objs):
            cls_id = self.cat_ids[annos['labels'][k]]
            bbox = annos['boxes'][k]
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = self.gaussian_radius((np.ceil(h), np.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                hm[cls_id] = self.draw_umich_gaussian(hm[cls_id], ct_int, radius)

        return hm

    def create_fixated_object_vector_target(self, img_name, fixation, output_h, output_w, fovea_size=16):
        """Create the target vector for object recognition at a fixated location."""
        annos = self.get_annos_by_img_name(img_name, output_h, output_w)
        num_objs = len(annos['boxes'])
        target = np.zeros(self.num_classes, dtype=np.float32)
        fovea_box = {
            'x1': max(0, fixation[0] - fovea_size),
            'y1': max(0, fixation[1] - fovea_size),
            'x2': min(output_w, fixation[0] + fovea_size),
            'y2': min(output_w, fixation[1] + fovea_size),
        }
        for k in range(num_objs):
            cls_id = self.cat_ids[annos['labels'][k]]
            bbox = annos['boxes'][k]
            iom = get_IoM(fovea_box, dict(zip(['x1', 'y1', 'x2', 'y2'], bbox)))
            target[cls_id] = iom
        if target.sum() > 0:
            target /= target.sum()
        return target
            
    def create_foveated_centermap_target(self, img_name, output_h, output_w, blur_slope):
        annos = self.get_annos_by_img_name(img_name, output_h, output_w)
        hms = []
        for i in range(5):
            hm = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32) + i * blur_slope
            num_objs = len(annos['boxes'])

            gt_det = []
            for k in range(num_objs):
                cls_id = self.cat_ids[annos['labels'][k]]
                bbox = annos['boxes'][k]
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                if h > 0 and w > 0:
                    radius = self.gaussian_radius((np.ceil(h), np.ceil(w)))
                    radius = max(0, int(radius))
                    ct = np.array(
                        [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                    ct_int = ct.astype(np.int32)
                    hm[cls_id] = self.draw_umich_gaussian(hm[cls_id], ct_int, radius, k=1-i*blur_slope)
            hms.append(hm)
        return np.concatenate(hms)
