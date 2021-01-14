"""
Various objects to use while training WSDDN
"""

import cv2 as cv
import numpy as np
from collections import OrderedDict

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

class EdgeBoxes():
    """Class to run OpenCV's EdgeBoxes impl."""
    def __init__(self, model_path, max_boxes=30):
        """Initialize the object

        Args:
            model_path (string): The path to the tarred model file (https://github.com/opencv/opencv_extra/blob/master/testdata/cv/ximgproc/model.yml.gz)
            max_boxes (int, optional): Max number of boxes to return. Defaults to 30.
        """
        self.edge_detection = cv.ximgproc.createStructuredEdgeDetection(model_path)
        self.edge_boxes = cv.ximgproc.createEdgeBoxes()
        self.edge_boxes.setMaxBoxes(max_boxes)

    def __call__(self, image, ret_scores=False):
        """Run EdgeBoxes on an image

        Args:
            image (Either PIL or tensor): The image
            ret_scores (bool, optional): If true, also returns the scores computed by the algorithm. Defaults to False.

        Returns:
            np.array: nx4 array with coordinates of the bouning box, formatted as xmin, ymin, xmax, ymax
        """
        cv_img = image
        if not isinstance(image, torch.Tensor):
            cv_img = transforms.PILToTensor()(cv_img)
        cv_img = cv_img.permute((1,2,0)).numpy()
        rgb_im = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        edges = self.edge_detection.detectEdges(np.float32(rgb_im) / 255.0)
        orimap = self.edge_detection.computeOrientation(edges)
        edges = self.edge_detection.edgesNms(edges, orimap)
        boxes = self.edge_boxes.getBoundingBoxes(edges, orimap)

        # OpenCV returns (x, y, w, h). Shift to be (x1, y1, x2, y2)
        for b in boxes[0]:
            b[2] += b[0]
            b[3] += b[1]
            
        if ret_scores:
            return boxes[0], boxes[1]
        
        return boxes[0]

class CustomVOC(datasets.VOCDetection):
    """Custom Dataset that wraps the default VOC one, also returns proposed bounding boxes"""
    def __init__(self, edge_model, eval_mode=False, new_boxes=False, augmentation=None, post_transform=None, *args, **kwargs):
        """Create the

        Args:
            edge_model (string): Path to the edge boxes model
            eval_mode (bool, optional): If true, also returns the ground truth boxes. Defaults to False.
            new_boxes (bool, optional): If true, runs the edge boxes algorithm on the image. Use this if the image is not from the train or val sets. Defaults to False.
            augmentation (transform, optional): An augmentation to apply to both the image and the bounding boxes. Defaults to None.
        """
        super().__init__(*args, **kwargs)
        self.eval_mode = eval_mode
        self.augmentation = augmentation
        self.post_transform = post_transform
        if new_boxes:
            self.edge_boxes = EdgeBoxes(edge_model, max_boxes=20)
        else:
            self.edge_boxes = None

        self.label_names = [
                    'background',
                    'aeroplane',
                    'bicycle',
                    'bird',
                    'boat',
                    'bottle',
                    'bus',
                    'car',
                    'cat',
                    'chair',
                    'cow',
                    'diningtable',
                    'dog',
                    'horse',
                    'motorbike',
                    'person',
                    'pottedplant',
                    'sheep',
                    'sofa',
                    'train',
                    'tvmonitor'
                ]

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        
        if self.edge_boxes is None:
            edge_boxes = label['annotation']['edge_box']['bndbox']
            boxes = np.zeros((len(edge_boxes), 4))
            for i, box in enumerate(edge_boxes):
                boxes[i] = np.array([box['xmin'], box['ymin'], box['xmax'], box['ymax']])
        else:
            boxes = self.edge_boxes(image)

        boxes = transforms.ToTensor()(boxes.astype('float32')).view([len(boxes), 4])
        
        if self.augmentation is not None:
            image, boxes = self.augmentation(image, boxes)
            
        if self.post_transform is not None:
            image = self.post_transform(image)

        one_hot_label = torch.zeros(len(self.label_names))
        # one_hot_label = torch.ones(len(self.label_names)) * -1
        objects = label['annotation']['object']
        if not isinstance(objects, list):
            objects = [objects]
            
        for obj in objects:
            one_hot_label[self.label_names.index(obj['name'])] = 1

        if torch.sum(one_hot_label) == -len(self.label_names):
            # Background
            one_hot_label[0] = 1


        if self.eval_mode:
            gt_boxes = []
            for obj in objects:
                label_index = self.label_names.index(obj['name'])
                bbox = obj['bndbox']
                gt_boxes.append([
                    int(bbox['xmin']), 
                    int(bbox['ymin']), 
                    int(bbox['xmax']), 
                    int(bbox['ymax']), 
                    label_index, 
                    int(obj['difficult']), 
                    0
                ])

            return image, one_hot_label, boxes, gt_boxes

        return image, one_hot_label, boxes


    def format_pred(self, net_detections):
        """Formats the detections so that they can be plotted by `draw_boxes`, as well as so they can be passed into a mAP calculation
        TODO: This should be its own function, no reason to be in this class

        Args:
            net_detections (dict): The output of `WSDNN.detect()`

        Returns:
            list: 6xn list with the 4 bbox coordinates, the class id, and the confidence
        """
        boxes = []
        for label in net_detections.keys():
            for box in net_detections[label]:
                boxes.append([*box[1], label, box[0]])
        
        return boxes


    def draw_boxes(self, image, pred_boxes, gt_boxes, thresh=0.6):
        """Draws the predictions in red and the gt in green

        Args:
            image (PIL.Image): The image to draw on
            pred_boxes (list): Output of `format_pred`: [xmin, ymin, xmax, ymax, class_id, confidence]
            gt_boxes (list): Output of __getitem__, with eval_mode set to True:[xmin, ymin, xmax, ymax, class_id, difficult, crowd]
            thresh (float, optional): Only draws predictions with confidence higher than this value. Defaults to 0.6.

        Returns:
            PIL.Image: The annotated image
        """

        
        
        img = image
        if not isinstance(image, torch.Tensor):
            img = transforms.PILToTensor()(image)

        img = img.permute((1,2,0)).numpy()

        for b in gt_boxes:
            x1, y1, x2, y2, class_id, _, _ = b
            img = cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1, cv.LINE_AA)
            label = self.label_names[class_id]
            img = self.draw_label(img, label, x1, y1, (0, 255, 0))
            
        for b in pred_boxes:
            x1, y1, x2, y2, class_id, conf = b
            if conf < thresh:
                continue
            img = cv.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1, cv.LINE_AA)
            label = self.label_names[class_id]
            img = self.draw_label(img, label, x1, y1, (255, 0, 0))


        return Image.fromarray(img)
    
    def draw_label(self, img, label, x1, y1, color):
        """Helper function to put text with a colored background

        Args:
            img (tensor/numpy array): The image to annotate
            label (string): The class name
            x1 (int): x coordinate for the bottom left of text
            y1 (int): y coordinate for the bottom left of text
            color ((R,G,B)): The color to draw the background rectange in

        Returns:
            numpy array: The annotated image
        """
        (label_width, label_height), baseline = cv.getTextSize(label, cv.FONT_HERSHEY_COMPLEX, 0.4, 1)
        back_tl = int(x1), int(y1 - int(1.3 * label_height))
        back_br = int(x1 + label_width), int(y1)
        cv.rectangle(img, back_tl, back_br, color, -1)
        txt_tl = int(x1), int(y1 - int(0.3 * label_height))
        return cv.putText(img, label, txt_tl, cv.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0))


    def draw_raw_boxes(self, image, boxes):
        """Simply draws the boxes in the image

        Args:
            image (PIL.Image or tensor): Image to draw on
            boxes (4xN): list of boxes

        Returns:
            PIL.Image: Annotated image
        """
        cv_img_drawn = image
        if not isinstance(image, torch.Tensor):
            try:
                cv_img_drawn = transforms.PILToTensor()(image)
            except:
                cv_img_drawn = transforms.ToTensor()(image)
                
        cv_img_drawn = cv_img_drawn.permute((1,2,0)).numpy()
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = b
            cv.rectangle(cv_img_drawn, (x1, y1), (x2, y2), (0, 255, 0), 1, cv.LINE_AA)

        return Image.fromarray(cv_img_drawn)


class BoxAndImageFlip(object):
    """Custom class to randomly flip an image and its bouning boxes"""
    def __init__(self, p_horiz=0.5, p_vert=0.1):
        """Create the objext

        Args:
            p_horiz (float, optional): Probability of flipping horizontally. Defaults to 0.5.
            p_vert (float, optional): TODO. Defaults to 0.1.
        """
        self.p_horiz = p_horiz
        self.p_vert = p_vert
    
    def __call__(self, image, boxes):
        if np.random.rand() < self.p_horiz:
            if isinstance(image, torch.Tensor):
                img_center = torch.tensor(image.shape[1:]).flip(0) / 2.
            else:
                # PIL
                img_center = torch.tensor(image.size) / 2.

            img_center = torch.tensor(np.hstack((img_center, img_center)))
            boxes[:, [0, 2]] += 2*(img_center[[0, 2]] - boxes[:, [0, 2]])
            box_w = torch.abs(boxes[:, 0] - boxes[:, 2])
            boxes[:, 0] -= box_w
            boxes[:, 2] += box_w

            image = TF.hflip(image)
        
        return image, boxes

class WSDDN(nn.Module):
    """Weakly supervised deep detection network"""
    def __init__(self, num_classes):
        super().__init__()
        vgg_backbone = torchvision.models.vgg16(pretrained=True)
        self.features = vgg_backbone.features[:-1]
        self.fcs = vgg_backbone.classifier[:-1] # 25088 -> 4096
        # Class level (_c) and detection level (_d) layers
        self.fc_c = nn.Linear(4096, num_classes)
        self.fc_d = nn.Linear(4096, num_classes)


    def forward(self, x, boxes):
        """Forward pass

        Args:
            x (tensor): image
            boxes (tensor): [Nx4] tensor of proposed regions

        Returns:
            tensor, FC7: 21xN tensor: class scores for each proposed region and FC7 output
        """
        x = self.features(x)
        x = torchvision.ops.roi_pool(x, [boxes], (7,7), 1.0 / 16)
        x = x.view(x.shape[0], -1)

        x = self.fcs(x)
        
        x_c = F.softmax(self.fc_c(x), dim=1)
        x_d = F.softmax(self.fc_d(x), dim=0)

        scores = x_c * x_d
        
        # Return final score and FC7 output, for regularization purposes
        return scores, x

    def detect(self, x, boxes):
        """Runs a forward pass and then NMS on the output. Groups the output by class

        Args:
            x (tensor): Predicted scores per region per class
            boxes (tensor): [Nx4] proposed regions

        Returns:
            dict: regions for each class
        """
        scores, _ = self.forward(x, boxes)

        num_boxes = len(boxes[0])
        num_classes = scores.shape[1]
        ranked_box_list = [OrderedDict() for _ in range(num_classes)]
        r_c_scores = torch.argmax(scores, dim=1)
        for i in range(num_boxes):
            score_idx = r_c_scores[i]
            ranked_box_list[score_idx][float(scores[i, score_idx].detach().cpu().numpy())] = list(boxes[i].cpu().numpy())

        predict_boxes = {}
        for cls_index, ranked_boxes in enumerate(ranked_box_list):
            if len(ranked_boxes) == 0:
                continue
            scores = np.array([list(ranked_boxes.keys())]).T
            bboxes = np.array(list(ranked_boxes.values()))
            tensor_boxes = transforms.ToTensor()(bboxes.astype('float')).view([len(scores), 4])
            tensor_scores = transforms.ToTensor()(scores.astype('float')).view(-1)
            keep = torchvision.ops.nms(tensor_boxes, tensor_scores, 0.4).cpu().numpy()
            keep_boxes = []
            ranked_boxes_items = list(ranked_boxes.items())
            for i in keep:
                keep_boxes.append(ranked_boxes_items[i])
            predict_boxes[cls_index] = keep_boxes
        
        return predict_boxes

        
    @staticmethod
    def loss(combined_scores, target):
        image_level_scores = torch.sum(combined_scores, dim=0)
        image_level_scores = torch.clamp(image_level_scores, min=0.0, max=1.0)
        loss = F.binary_cross_entropy(image_level_scores, target, reduction="sum")
        return loss
    
    @staticmethod
    def reg(scores, fc7, boxes):
        class_scores = torch.argmax(scores, dim=0)
        num_classes = class_scores.shape[0]
        reg = 0
        for k in range(num_classes):
            # For each class, grab the highest scoring region
            highest_region = class_scores[k]
            highest_score = scores[highest_region, k]
            # Get any regions with IOU > 0.6
            overlap_idxs = (torchvision.ops.box_iou(boxes[highest_region].reshape(1, 4), boxes) > 0.7)[0]
            for r in list(torch.where(overlap_idxs)[0]):
                # For each region in the high overlapping ones
                diff = fc7[highest_region] - fc7[r]
                reg += torch.pow(highest_score, 2) * torch.matmul(diff.T, diff) / 2
        
        return reg / num_classes

    # def spatial_pyramid_pool(self, previous_conv, num_sample, previous_conv_size, out_pool_size):
    #     # Source: https://github.com/yueruchen/sppnet-pytorch/blob/master/cnn_with_spp.py
    #     '''
    #     previous_conv: a tensor vector of previous convolution layer
    #     num_sample: an int number of image in the batch
    #     previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    #     out_pool_size: a int vector of expected output size of max pooling layer
        
    #     returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    #     '''    
    #     # print(previous_conv.size())
    #     for i in range(len(out_pool_size)):
    #         # print(previous_conv_size)
    #         h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
    #         w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
    #         h_pad = (h_wid*out_pool_size[i] - previous_conv_size[0] + 1)/2
    #         w_pad = (w_wid*out_pool_size[i] - previous_conv_size[1] + 1)/2
    #         maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
    #         x = maxpool(previous_conv)
    #         if (i == 0):
    #             spp = x.view(num_sample,-1)
    #             # print("spp size:",spp.size())
    #         else:
    #             # print("size:",spp.size())
    #             spp = torch.cat((spp,x.view(num_sample,-1)), 1)
        
    #     return spp