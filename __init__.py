import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import uuid
import os

from model import U2NET
from torch.autograd import Variable
from skimage import io, transform
from PIL import Image

# Get the current directory
currentDir = os.path.dirname(__file__)

# Functions:
# Save results


def save_output(image_name, output_name, pred, d_dir, type):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np * 255).convert('RGB')
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]))
    pb_np = np.array(imo)
    if type == 'image':
        # Make and apply mask
        mask = pb_np[:, :, 0]
        mask = np.expand_dims(mask, axis=2)
        imo = np.concatenate((image, mask), axis=2)
        imo = Image.fromarray(imo, 'RGBA')

    imo.save(os.path.join(d_dir, output_name))


# Remove background from image (generate mask and final results)
def remove_bg(image_path):
    inputs_dir = os.path.join(currentDir, 'static', 'inputs')
    results_dir = os.path.join(currentDir, 'static', 'results')
    masks_dir = os.path.join(currentDir, 'static', 'masks')

    # Convert string of image data to uint8
    with open(image_path, "rb") as image:
        f = image.read()
        img = bytearray(f)

    nparr = np.frombuffer(img, np.uint8)

    if len(nparr) == 0:
        return '---Empty image---'

    # Decode image
    try:
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except:
        # Build a response dict to send back to the client
        return "---Empty image---"

    # Save image to inputs
    unique_filename = str(uuid.uuid4())
    cv2.imwrite(os.path.join(inputs_dir, unique_filename + '.jpg'), img)

    # Processing
    image = transform.resize(img, (320, 320), mode='constant')

    tmp_img = np.zeros((image.shape[0], image.shape[1], 3))

    tmp_img[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
    tmp_img[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
    tmp_img[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

    tmp_img = tmp_img.transpose((2, 0, 1))
    tmp_img = np.expand_dims(tmp_img, 0)
    image = torch.from_numpy(tmp_img)

    if torch.cuda.is_available():
        image = image.type(torch.FloatTensor).cuda()
    else:
        image = image.type(torch.FloatTensor)

    image = Variable(image)

    d1, d2, d3, d4, d5, d6, d7 = net(image)
    pred = d1[:, 0, :, :]
    ma = torch.max(pred)
    mi = torch.min(pred)
    dn = (pred - mi) / (ma - mi)
    pred = dn

    save_output(os.path.join(inputs_dir, unique_filename + '.jpg'),
                unique_filename + '.png', pred, results_dir, 'image')
    save_output(os.path.join(inputs_dir, unique_filename + '.jpg'),
                unique_filename + '.png', pred, masks_dir, 'mask')

    return "---Success---"


# ------- Load Trained Model --------
print("---Loading Model---")
model_name = 'u2net'
model_dir = os.path.join(currentDir, 'saved_models', model_name, model_name + '.pth')
net = U2NET(3, 1)

if torch.cuda.is_available():
    net.load_state_dict(torch.load(model_dir))
    net.cuda()
else:
    net.load_state_dict(torch.load(model_dir, map_location='cpu'))
# ------- Load Trained Model --------

print("---Removing Background...")
# ------- Call the remove_bg Function --------
img_path = "1.jpg"  # Change this to your image path
print(remove_bg(img_path))
