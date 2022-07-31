import numpy as np
import torch

import solar_loss
from solar_dataset import Solar_Dataset, Solar_Dataset_wo_label
import solar_model
from torchvision import transforms
from torch.utils.data import DataLoader

from torch.optim import Adam
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import cv2

# model save path
mode_save_path = r'save_model/checkpoint.pt'

# pretrain model weight
load_model = False
model_load_path = r'save_model/checkpoint.pt'

# define the test split
test_split = 0.2
# determine the device to be used for training and evaluation
device = "cuda" if torch.cuda.is_available() else "cpu"
# determine if we will be pinning memory during data loading
pin_memory = True if device == "cuda" else False
# determine num workers
num_worker = 0

# define the number of channels in the input, number of classes
num_channel = 1
num_classes = 1

# initialize learning rate, number of epochs to train for, and the batch size
lr = 1e-3
num_epochs = 200
batch_size = 8

# define the input image dimensions
width, height = 512, 512

# define threshold to filter weak predictions
threshold = 0.5

# dataset
train_img_path = r'F:\solar_IR\SAMPLED\label_img\train'
valid_img_path = r'F:\solar_IR\SAMPLED\label_img\validation'
test_img_path = r'F:\solar_IR\SAMPLED\label_img\test'
pred_img_path = r'F:\solar_IR'

transform_input = transforms.Compose([transforms.ToPILImage(),
                                      transforms.RandomRotation(90),
                                      # transforms.RandomCrop(256),
                                      transforms.Resize((width, height)),
                                      transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                      transforms.ToTensor(),
                                      ])
transform_label = transforms.Compose([transforms.ToPILImage(),
                                      transforms.RandomRotation(90),
                                      # transforms.RandomCrop(256),
                                      transforms.Resize((width, height)),
                                      transforms.ToTensor(),
                                      ])
transform_pred = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize((width, height)),
                                     transforms.ToTensor(),
                                     ])

trainDS = Solar_Dataset(src_path=train_img_path, transforms_input=transform_input, transform_label=transform_label)
validDS = Solar_Dataset(src_path=valid_img_path, transforms_input=transform_input, transform_label=transform_label)
testDS = Solar_Dataset(src_path=test_img_path, transforms_input=transform_input, transform_label=transform_label)
predDS = Solar_Dataset_wo_label(src_path=pred_img_path, transforms=transform_pred)
print(f'Train data: {len(trainDS)}, Validation data: {len(validDS)}, Test data: {len(testDS)}, Pred data: {len(predDS)}')

trainLoader = DataLoader(trainDS, shuffle=True,
                         batch_size=batch_size, pin_memory=pin_memory,
                         num_workers=num_worker)
validationLoader = DataLoader(validDS, shuffle=False,
                              batch_size=batch_size, pin_memory=pin_memory,
                              num_workers=num_worker)
testLoader = DataLoader(testDS, shuffle=False,
                        batch_size=batch_size, pin_memory=pin_memory,
                        num_workers=num_worker)
predLoader = DataLoader(predDS, shuffle=False,
                        batch_size=batch_size, pin_memory=pin_memory,
                        num_workers=num_worker)

# setting model
net = solar_model.UNet(num_channel, num_classes)
net.to(device)

# setting loss func and optimizer
criterion = solar_loss.IoULoss()
optimizer = Adam(net.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)


# load pretrain model weight
if load_model:
    checkpoint = torch.load(model_load_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']


# eval
def model_eval(model, loader, dev):
    valid_loss = 0.0
    model.eval()

    with tqdm(loader, unit="batch") as t_epoch:
        for val_data in t_epoch:
            # get the inputs; data is a list of [inputs, labels]
            with torch.no_grad():
                valid_inputs, valid_labels = val_data
                valid_inputs = valid_inputs.to(dev)
                valid_labels = valid_labels.to(dev)

                # forward + backward + optimize
                valid_pred = model(valid_inputs)
                batch_loss = criterion(valid_pred, valid_labels)
                valid_loss += batch_loss
                tepoch.set_postfix(loss=batch_loss.item())

        # print(predictions, val_labels)
        # print(max_predictions, restore_labels)
        avg_loss = valid_loss / len(loader)
        return avg_loss, valid_inputs, valid_labels, valid_pred


writer = SummaryWriter()
best_val = 1.0
for epoch in range(num_epochs):
    epoch_loss = 0.0

    with tqdm(trainLoader, unit="batch") as tepoch:
        for data in tepoch:
            net.train()
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss

            tepoch.set_postfix(loss=loss.item())

        print('Training: epoch:', epoch, 'loss:', epoch_loss.item() / len(trainLoader))

        writer.add_scalar('train_loss', epoch_loss.item() / len(trainLoader), epoch)
        writer.add_images('train_image', inputs, epoch)
        writer.add_images('train_label', labels, epoch)
        writer.add_images('train_pred', outputs, epoch)

        avg_val_loss, val_inputs, val_labels, val_pred = model_eval(net, validationLoader, device)
        writer.add_images('val_image', val_inputs, epoch)
        writer.add_images('val_label', val_labels, epoch)
        writer.add_images('val_pred', val_pred, epoch)
        writer.add_scalar('val_loss', avg_val_loss, epoch)
        print('Validation: epoch:', epoch, 'loss:', avg_val_loss.item())
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            print('New best! Saving...')
            torch.save(dict(epoch=epoch, model_state_dict=net.state_dict(), optimizer_state_dict=optimizer.state_dict(), loss=loss), mode_save_path)
        scheduler.step(avg_val_loss)

writer.close()

test_loss, test_inputs, test_labels, test_pred = model_eval(net, testLoader, device)
print('Test loss: ', test_loss.item())


# Pred
def model_pred(model, loader, dev):
    model.eval()

    with tqdm(loader, unit="batch") as t_epoch:
        for pred_data in t_epoch:
            with torch.no_grad():
                pred_inputs, pred_path = pred_data
                pred_inputs = pred_inputs.to(dev)
                try:
                    pred_pred = model(pred_inputs)
                    pred_pred = pred_pred.cpu()
                    pred_inputs = pred_inputs.cpu()
                    # print(pred_pred.shape)
                    for idx in range(pred_pred.shape[0]):
                        mask = np.array(pred_pred[idx, :, :, :])
                        img = np.array(pred_inputs[idx, :, :, :])
                        img = np.multiply(mask, img)
                        img = img[0, :, :]
                        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        img = img.astype(np.uint8)

                        img_path = pred_path[idx]
                        save_name = img_path[:-4] + '_masked.jpg'

                        cv2.imwrite(save_name, img)
                except Exception as e:
                    print(e)
                    print(pred_path)


model_pred(net, predLoader, device)
