import os
from xml.etree import ElementTree as et
import torch
import shutil

path = r'F:\solar_RGB\Sampled\devided_merge'
mv_path = r'F:\solar_RGB\Sampled\divided_empty'

lack_ls = []
for dp, dn, fp in os.walk(path):
    for f in fp:
        if '.jpg' or '.JPG' in f:
            if not os.path.exists(os.path.join(dp, f[:-4]+'.xml')):
                lack_ls.append(os.path.join(dp, f))

print(lack_ls)

src_list = []
area_ls = []
for dp, dn, fp in os.walk(path):
    for f in fp:
        if '.xml' in f:
            boxes = []
            xml_path = os.path.join(dp, f)
            tree = et.parse(xml_path)
            root = tree.getroot()

            for member in root.findall('object'):
                # map the current object name to `classes` list to get...
                # ... the label index and append to `labels` list

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

                boxes.append([xmin, ymin, xmax, ymax])

            # bounding box to tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # area of the bounding boxes
            # print(torch.Tensor.dim(boxes))
            # print(boxes)
            if torch.Tensor.dim(boxes) > 1:
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                for a in area:
                    area_ls.append(a)


total_area = 0
for a in area_ls:
    total_area += a

print('avg area:', total_area/len(area_ls))
