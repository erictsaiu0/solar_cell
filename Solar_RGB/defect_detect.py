import torch
import cv2
import numpy as np
import os
from xml.etree import ElementTree as et

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.models.detection
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import FPN_transforms as T

from FPN_engine import train_one_epoch, evaluate
import utils
import random

import matplotlib.pyplot as plt

src_dir = r'F:\solar_RGB\Sampled\labeled\devided_merge'
test_dir = r'F:\solar_RGB\Sampled\unlabeled_merged'

load_model = True
model_save_path = r'C:\Users\EricTsai\PycharmProjects\solar_cell\Solar_RGB\model\model_300.pkl'

input_size = 1024
batch_size = 16
num_classes = 2  # background and defects
classes = ['background', 'defect']

num_epochs = 0

# seed setup
seed = 7085
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# dataset and transforms
class DefectDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None, testing=False):
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes
        self.testing = testing

        # get all the image paths in sorted order
        self.image_paths = glob.glob(f"{self.dir_path}/*.jpg")
        self.all_images = [image_path.split('/')[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        # capture the image name and the full image path
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name)

        # read the image
        image = cv2.imread(image_path)
        # convert BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        # capture the corresponding XML file for getting the annotations
        annot_filename = image_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.dir_path, annot_filename)

        # get the height and width of the image
        image_width = image.shape[1]
        image_height = image.shape[0]

        boxes = []
        labels = []
        if not self.testing:
            tree = et.parse(annot_file_path)
            root = tree.getroot()

            # box coordinates for xml files are extracted and corrected for image size given
            for member in root.findall('object'):
                # map the current object name to `classes` list to get...
                # ... the label index and append to `labels` list
                labels.append(self.classes.index(member.find('name').text))

                # xmin = left corner x-coordinates
                xmin = int(member.find('bndbox').find('xmin').text)
                # xmax = right corner x-coordinates
                xmax = int(member.find('bndbox').find('xmax').text)
                # ymin = left corner y-coordinates
                ymin = int(member.find('bndbox').find('ymin').text)
                # ymax = right corner y-coordinates
                ymax = int(member.find('bndbox').find('ymax').text)

                # resize the bounding boxes according to the...
                # ... desired `width`, `height`
                xmin_final = (xmin / image_width) * self.width
                xmax_final = (xmax / image_width) * self.width
                ymin_final = (ymin / image_height) * self.height
                yamx_final = (ymax / image_height) * self.height

                boxes.append([xmin_final, ymin_final, xmax_final, yamx_final])

            # bounding box to tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # area of the bounding boxes
            # print(torch.Tensor.dim(boxes))
            # print(boxes)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # no crowd instances
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
            # labels to tensor
            labels = torch.as_tensor(labels, dtype=torch.int64)

        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            area = 0
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        # apply the image transforms
        if self.transforms:
            sample = self.transforms(image=image_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return image_resized, target

    def __len__(self):
        return len(self.all_images)


train_transform = A.Compose([
        A.Flip(p=0.5),
        A.RandomRotate90(p=0.5),
        # A.MotionBlur(p=0.1),
        # A.MedianBlur(blur_limit=3, p=0.1),
        # A.Blur(blur_limit=3, p=0.1),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })
val_transform = A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

# prepare the final datasets and data loaders
total_dataset = DefectDataset(src_dir, input_size, input_size, classes, train_transform)
test_dataset = DefectDataset(test_dir, input_size, input_size, classes, val_transform, testing=True)

indices = torch.randperm(len(total_dataset)).tolist()

split_ratio = 0.8
train_len = int(split_ratio*len(total_dataset))
val_len = len(total_dataset)-train_len

train_dataset, valid_dataset = torch.utils.data.random_split(total_dataset, [train_len, val_len])

print('Train data:', len(train_dataset), 'Validation data:', len(valid_dataset), 'Test data:', len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True, collate_fn=utils.collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False, collate_fn=utils.collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False, collate_fn=utils.collate_fn)

# create model
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#
# in_features = model.roi_heads.box_predictor.cls_score.in_features
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)

if load_model:
    model = torch.load(model_save_path)

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]

# SGD
optimizer = torch.optim.Adam(params, lr=1e-3)

# and a learning rate scheduler
# cos學習率
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

best_loss = 1.0

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    # engine.py的train_one_epoch函式將images和targets都.to(device)了
    logger, losses = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=50)

    print(f'\nCurrent best: {best_loss}, loss: {losses}\n')

    if losses < best_loss:
        print(f'New Best! Saving...\n')
        best_loss = losses
        torch.save(model, model_save_path)

        print('Test on validation')
        # evaluate on the test dataset
        val_eval = evaluate(model, valid_loader, device=device)

    if (epoch+1) % 10 == 0:
        print('Test on Train')
        evaluate(model, train_loader, device=device)

    # update the learning rate
    lr_scheduler.step(losses)

    print('')
    print('==================================================')
    print('')

if num_epochs > 0:
    print("End training!")


def showbbox(model, img, target, save_name=None):
    # 輸入的img是0-1范圍的tensor
    model.eval()
    with torch.no_grad():
        '''
        prediction形如：
        [{'boxes': tensor([[1492.6672,  238.4670, 1765.5385,  315.0320],
        [ 887.1390,  256.8106, 1154.6687,  330.2953]], device='cuda:0'), 
        'labels': tensor([1, 1], device='cuda:0'), 
        'scores': tensor([1.0000, 1.0000], device='cuda:0')}]
        '''
        prediction = model([img.to(device)])
    target_bbox = target["boxes"].cpu()
    pred_bbox = prediction[0]['boxes'].cpu()
    IoU = utils.bbox_overlaps(pred_bbox, target_bbox)
    print(f'Test IoU: {IoU}')

    if save_name:

        imgA = img.permute(1, 2, 0)  # C,H,W → H,W,C，用來畫圖
        imgA = (imgA * 255).byte().data.cpu()  # * 255，float轉0-255
        imgA = np.array(imgA)  # tensor → ndarray

        for i in range(prediction[0]['boxes'].cpu().shape[0]):
            xmin = round(prediction[0]['boxes'][i][0].item())
            ymin = round(prediction[0]['boxes'][i][1].item())
            xmax = round(prediction[0]['boxes'][i][2].item())
            ymax = round(prediction[0]['boxes'][i][3].item())

            label = prediction[0]['labels'][i].item()

            if label == 1:
                cv2.rectangle(imgA, (xmin, ymin), (xmax, ymax), (255, 0, 0), thickness=2)
                cv2.putText(imgA, 'defect', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                            thickness=2)

        img_lbl = img.permute(1, 2, 0)  # C,H,W → H,W,C，用來畫圖
        img_lbl = (img_lbl * 255).byte().data.cpu()  # * 255，float轉0-255
        img_lbl = np.array(img_lbl)  # tensor → ndarray

        for i in range(target['boxes'].cpu().shape[0]):
            xmin = round(target['boxes'][i][0].item())
            ymin = round(target['boxes'][i][1].item())
            xmax = round(target['boxes'][i][2].item())
            ymax = round(target['boxes'][i][3].item())

            label = target['labels'][i].item()

            if label == 1:
                cv2.rectangle(img_lbl, (xmin, ymin), (xmax, ymax), (255, 0, 0), thickness=2)
                cv2.putText(img_lbl, 'defect', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),
                            thickness=2)

        final_img = np.concatenate((imgA, img_lbl), axis=1)

        plt.figure(figsize=(20, 15))
        plt.imshow(final_img)
        plt.tight_layout()
        plt.savefig(save_name, dpi=300)
        plt.close()


model = torch.load(model_save_path)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

print('\nTesting and saving results...\n')
result_save_path = r'C:\Users\EricTsai\PycharmProjects\solar_cell\Solar_RGB\result_500_test'
if not os.path.exists(result_save_path):
    os.mkdir(result_save_path)


for i in range(len(valid_dataset)):
    img, lbl = valid_dataset[i]
    save_path = os.path.join(result_save_path, 'result_'+str(i)+'.png')
    showbbox(model, img, lbl)

# for i in range(len(test_dataset)):
#     img, lbl = test_dataset[i]
#     save_path = os.path.join(result_save_path, 'result_'+str(i)+'.png')
#     showbbox(model, img, lbl, save_path)
