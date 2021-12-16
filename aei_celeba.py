import os
import csv
import argparse
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
from torchvision import transforms

from aei_net import AEINet

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, default="config/train.yaml",
                    help="path of configuration yaml file")
parser.add_argument("--checkpoint_path", type=str, required=True,
                    help="path of aei-net pre-trained file")
parser.add_argument("--gpu_num", type=int, default=0,
                    help="number of gpu")
args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else 'cpu')

hp = OmegaConf.load(args.config)
model = AEINet.load_from_checkpoint(args.checkpoint_path, hp=hp)
model.eval()
model.freeze()
model.to(device)

transfs = {
    "32": transforms.Compose([transforms.Resize(32),
                                transforms.Resize(256),
                                transforms.ToTensor()]),
    "64": transforms.Compose([transforms.Resize(64),
                                transforms.Resize(256),
                                transforms.ToTensor()]),
    "orig": transforms.Compose([transforms.Resize(256),
                                transforms.ToTensor()])
}

base_imgs = "/DATA/data512x512"
base_out = "/DATA/faceshifter_output"

for exp in ["orig", "64", "32"]:
    img_path = base_imgs #+ str(exp) #os.path.join(base_imgs, str(exp))
    output_path = os.path.join(base_out, str(exp))
    os.makedirs(output_path, exist_ok=True)

    pairs = open("./celebahq_test.csv", "r")
    pairs = list(csv.reader(pairs))[1:]
    # _ = next(pairs)

    pil2tensor_transform = transfs[str(exp)]
    pil2tensor_transform_src = transfs["orig"]

    for (src, trg) in tqdm(pairs[:1000]):
        src_p, trg_p = os.path.join(img_path, src), os.path.join(img_path, trg)
        if os.path.isfile(src_p) and os.path.isfile(trg_p):
            target_img = pil2tensor_transform(Image.open(args.target_image)).unsqueeze(0).to(device)
            source_img = pil2tensor_transform_src(Image.open(args.source_image)).unsqueeze(0).to(device)

            target_img = transforms.ToTensor()(Image.open(args.target_image)).unsqueeze(0).to(device)
            source_img = transforms.ToTensor()(Image.open(args.source_image)).unsqueeze(0).to(device)
            with torch.no_grad():
                output, _, _, _, _ = model.forward(target_img, source_img)
            output = transforms.ToPILImage()(output.cpu().squeeze().clamp(0, 1))
            output.save(os.path.join(output_path, trg))