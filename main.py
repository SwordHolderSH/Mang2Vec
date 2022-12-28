# ! encoding:UTF-8
import time
import os
import cv2
import argparse
import point2svg_div as ps
from point2svg_div import Point2svg
from decode import Decode_np
import decode as d
import vectorize_utils as vu
import torch
from DRL.actor import ResNet
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Learning to Paint')
parser.add_argument('--max_step', default=40, type=int, help='max length for episode')
parser.add_argument('--actor', default='./model/actor.pkl', type=str, help='Actor model')
parser.add_argument('--img', default='./image/Naruto.png', type=str, help='test image')
parser.add_argument('--imgid', default=0, type=int, help='set begin number for generated image')
parser.add_argument('--divide', default=32, type=int, help='divide the target image to get better resolution')
parser.add_argument('--width', default=128, type=int, help='width of each patch')
parser.add_argument('--output_dir', default='./output/', type=str, help='output path')
args = parser.parse_args()

width = args.width
divide = args.divide
output_dir = args.output_dir
canvas_cnt = divide * divide
use_patch_fill = True
use_PM = False  # Whether to use the pruning module
T = torch.ones([1, 1, width, width], dtype=torch.float32).to(device)
coord = vu.get_coord(width=width, device=device)
d.del_file(output_dir)

if __name__ == '__main__':
    actor = ResNet(5, 18, 9)
    actor.load_state_dict(torch.load(args.actor))
    actor = actor.to(device).eval()

    img = cv2.imread(args.img, cv2.IMREAD_GRAYSCALE)
    img0 = img
    (h, w) = img.shape
    origin_shape = (img.shape[1], img.shape[0])

    if use_patch_fill is True:
        canvas, patch_done_list = ps.patch_fill(img=img, div_num=divide)
        canvas = cv2.resize(canvas, (width * args.divide, width * args.divide)).astype('float32')
        canvas = torch.from_numpy(canvas / 255)
        canvas = canvas.unsqueeze(0).unsqueeze(0).to(device)
    else:
        canvas = torch.ones([1, 1, width, width]).to(device)
        _, patch_done_list = ps.patch_fill(img=img, div_num=args.divide)

    patch_img = cv2.resize(img, (width * args.divide, width * args.divide))
    patch_img = vu.binarize(patch_img)
    patch_img = vu.gray_div(patch_img)
    r = h / w
    p = cv2.resize(patch_img, (int(width * args.divide), int(width * args.divide * r)))
    cv2.imwrite(filename=output_dir + 'target.png', img=p)
    patch_img = vu.large2small(patch_img, canvas_cnt, divide, width)
    patch_img = np.transpose(patch_img, (0, 3, 1, 2))
    patch_img = torch.tensor(patch_img).to(device).float() / 255.

    img = cv2.resize(img, (width, width))
    img = img.reshape(1, width, width, 1)
    img = np.transpose(img, (0, 3, 1, 2))
    img = torch.tensor(img).to(device).float() / 255.

    os.system('mkdir output')
    p2s = Point2svg(width=width, div_num=divide, save_path=output_dir, init_num=0, img_w=w, img_h=h, img=img0,
                    use_patch_fill=use_patch_fill, patch_done_list=patch_done_list)

    act_list = []

    with torch.no_grad():
        # if args.divide != 1:
        # args.max_step = args.max_step // 2
        if args.divide != 0:
            canvas = canvas[0].detach().cpu().numpy()  # (1, 128, 128)
            canvas = np.transpose(canvas, (1, 2, 0))  # (128, 128, 1)
            canvas = cv2.resize(canvas, (width * args.divide, width * args.divide))  # (4096, 4096)
            canvas = vu.large2small(canvas, canvas_cnt, divide, width)  # (1024, 128, 128, 1)
            canvas = np.transpose(canvas, (0, 3, 1, 2))  # (1024, 1, 128, 128)
            canvas = torch.tensor(canvas).to(device).float()  # torch.Size([1024, 1, 128, 128])
            coord = coord.expand(canvas_cnt, 2, width, width)  # torch.Size([1024, 2, 128, 128])
            T = T.expand(canvas_cnt, 1, width, width)  # torch.Size([128, 128, 2, 1024])
            vu.save_img(canvas, args.imgid, divide_number=divide, width=width, origin_shape=origin_shape, divide=True)
            args.imgid += 1
            start = time.time()
            for i in range(args.max_step):
                stepnum = T * i / args.max_step
                actions = actor(torch.cat([canvas, patch_img, stepnum, coord], 1))
                p2s.reset_gt_patch(gt=patch_img)
                canvas, res = vu.decode_list(actions, canvas)
                print('divided canvas step {}, Loss = {}'.format(i, ((canvas - patch_img) ** 2).mean()))
                p2s.add_action_div(actions)  # =================================
                vu.save_img(canvas, args.imgid, divide_number=divide, width=width, origin_shape=origin_shape,
                            divide=True)
                args.imgid += 1

            end1 = time.time()
            unless_time = p2s.draw_action_list_for_all_patch(path_or_circle='path')
            unless_time2_s = time.time()
            d = Decode_np(div_num=divide, use_PM=use_PM)
            unless_time2_e = time.time()
            d.draw_decode()
            end2 = time.time()

            time_actor = end1 - start
            time_paint = end2 - end1 - unless_time - (unless_time2_e - unless_time2_s)
            print("actor time is : {}".format(time_actor))
            print("paint time is : {}".format(time_paint))
