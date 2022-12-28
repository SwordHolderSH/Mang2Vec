# ! encoding:UTF-8

import numpy as np
import cv2
import torch
from PIL import Image
import cairosvg
import time


exitFlag = 0


def gray_div_01_tensor(x):
    x = torch.round(x * 10) / 10
    return x


def gray_div(img):
    img = img.astype('float32') / 255
    img = np.around(img, 1) * 255
    img = img.astype('uint8')
    return img


def get_current_png(svg_text, width, patch_num):
    tmp_path = './output/tmp_simply_{}.png'.format(patch_num)
    try:
        cairosvg.svg2png(bytestring=svg_text, write_to=tmp_path, output_width=width, output_height=width)
    except:
        try:
            time.sleep(0.00001)
            cairosvg.svg2png(bytestring=svg_text, write_to=tmp_path, output_width=width, output_height=width)
        except:
            pass
    current_png = cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE)
    current_png = gray_div(current_png)
    return current_png


def cal_dis(svg_text, gt, svg_s, svg_e, patch_num, width=128):
    svg_total = '''{}\n{}\n{}'''.format(svg_s, svg_text, svg_e)
    current_png = get_current_png(svg_total, width=128, patch_num=patch_num)
    dis = np.abs(current_png.astype('float32') - gt.astype('float32')).mean(0).mean(0) / width / width

    return dis, current_png


def simplify_patch_action(svg_text, width, color, gt_patch, patch_coord, coord_bais, div_num, patch_num):
    [i, j] = patch_coord
    [cx, cy] = coord_bais
    baisx = cx[i][j]
    baisy = cy[i][j]
    k = i + j * div_num
    vy0, vx0, vw, vh = baisx, baisy, width, width
    color = color[k][0]
    # svg_text = svg_text_total[patch_num]
    svg_s = '''<svg xmlns="http://www.w3.org/2000/svg" version="1.1" height="{}" width="{}" viewBox="{} {} {} {}">\n<rect x="{}" y="{}" width="{}" height="{}" fill="{}"/>\n'''.format(
        width, width, vx0, vy0, vw, vh, vx0, vy0, width, width, color)
    svg_e = '''</svg>'''
    gt = (np.round(gt_patch, 1) * 255).astype('uint8')
    last_dis, current_png = cal_dis(svg_text, gt, svg_s, svg_e, patch_num)
    num = 0
    # g_path = './output/test1/{}_{}_a.png'.format(k,  num)
    # g = Image.fromarray(gt).convert('L')
    # g.save(g_path)
    # c_path = './output/test1/{}_{}_c.png'.format(k, num)
    # c = Image.fromarray(current_png).convert('L')
    # c.save(c_path)
    flag = 'not_done'
    save_gt_and_init(gt, svg_text, patch_num, svg_s, svg_e)
    while (flag != 'done'):
        num = num + 1
        last_dis, svg_text, flag, current_png = delete_svg(last_dis, svg_text, gt, svg_s, svg_e, patch_num)
        # print(num, last_dis)
    svg_text = list2string(list=svg_text)
    return svg_text

def simplify_patch_actionv_no_PM(svg_text, width, color, gt_patch, patch_coord, coord_bais, div_num, patch_num):
    [i, j] = patch_coord
    [cx, cy] = coord_bais
    baisx = cx[i][j]
    baisy = cy[i][j]
    k = i + j * div_num
    vy0, vx0, vw, vh = baisx, baisy, width, width
    color = color[k][0]
    # svg_text = svg_text_total[patch_num]
    svg_s = '''<svg xmlns="http://www.w3.org/2000/svg" version="1.1" height="{}" width="{}" viewBox="{} {} {} {}">\n<rect x="{}" y="{}" width="{}" height="{}" fill="{}"/>\n'''.format(
        width, width, vx0, vy0, vw, vh, vx0, vy0, width, width, color)
    svg_e = '''</svg>'''
    gt = (np.round(gt_patch, 1) * 255).astype('uint8')
    last_dis, current_png = cal_dis(svg_text, gt, svg_s, svg_e, patch_num)
    num = 0
    # g_path = './output/test1/{}_{}_a.png'.format(k,  num)
    # g = Image.fromarray(gt).convert('L')
    # g.save(g_path)
    # c_path = './output/test1/{}_{}_c.png'.format(k, num)
    # c = Image.fromarray(current_png).convert('L')
    # c.save(c_path)
    flag = 'not_done'
    save_gt_and_init(gt, svg_text, patch_num, svg_s, svg_e)
    # while (flag != 'done'):
    #     num = num + 1
    #     last_dis, svg_text, flag, current_png = delete_svg(last_dis, svg_text, gt, svg_s, svg_e, patch_num)
    #     # print(num, last_dis)
    svg_text = list2string(list=svg_text)
    return svg_text

def list2string(list):
    s = ""
    for i in list:
        s = s + i
    return s


def delete_svg(dis_init, svg_text, gt, svg_s, svg_e, patch_num):
    flag = 'not_done'
    if len(svg_text) == 0:
        flag = 'done'
        svg_text = ['']
        dis, current_png = cal_dis(svg_text, gt, svg_s, svg_e, patch_num)
        return dis, svg_text, flag, current_png

    for s in range(len(svg_text)):
        i = len(svg_text) - 1 - s
        svg_text1 = svg_text.copy()
        svg_text1.pop(i)
        dis, current_png = cal_dis(svg_text1, gt, svg_s, svg_e, patch_num)
        if dis <= dis_init:
            svg_text = svg_text1
            flag = 'not_done'
            # print('delete == {}'.format(i))
            # c_path = './output/test1/{}_{}_c.png'.format(patch_num, i)
            # c = Image.fromarray(current_png).convert('L')
            # c.save(c_path)
            return dis, svg_text, flag, current_png
        elif s == len(svg_text) - 1 and dis > dis_init:
            flag = 'done'
            return dis, svg_text, flag, current_png
        elif s == len(svg_text) - 1 and dis <= dis_init:
            svg_text = svg_text1
            flag = 'done'
            return dis, svg_text, flag, current_png


def save_gt_and_init(gt, svg_text, patch_num, svg_s, svg_e, width=128):
    svg_text = '''{}\n{}\n{}'''.format(svg_s, svg_text, svg_e)

    gt_path = './output/tmp_simply_{}_gt.png'.format(patch_num)
    tmp_path = './output/tmp_simply_{}_init.png'.format(patch_num)

    gt = Image.fromarray(gt).convert('L')
    try:
        gt.save(gt_path)
        cairosvg.svg2png(bytestring=svg_text, write_to=tmp_path, output_width=width, output_height=width)
    except:
        try:
            time.sleep(0.00001)
            gt.save(gt_path)
            cairosvg.svg2png(bytestring=svg_text, write_to=tmp_path, output_width=width, output_height=width)
        except:
            pass


class myThread(object):
    '''The multithreading was canceled, since we found that it did not get faster.'''

    def __init__(self, config, patch_bgcolor_list, fill_list, use_PM=True):
        [self.svg_text, self.width, self.gt_patch, self.patch_coord,
         self.coord_bais, self.div_num, self.patch_num] = config
        self.svg_txt_one_patch_s = '''<g style="clip-path: url(#clipPath{}); ">'''.format(self.patch_num)
        self.svg_txt_one_patch_e = '''</g>'''
        self.color = patch_bgcolor_list
        self.svg_test_init = self.svg_text.copy()
        self.patch_fill = fill_list[self.patch_num][0]
        self.use_PM = use_PM

    def run(self):
        self.svg_text = simplify_patch_action(svg_text=self.svg_text, width=self.width,
                                                     color=self.color, gt_patch=self.gt_patch,
                                                     patch_coord=self.patch_coord, coord_bais=self.coord_bais,
                                                     div_num=self.div_num, patch_num=self.patch_num)
        self.svg_txt_one_patch = '''{}\n{}\n{}{}'''.format(self.svg_txt_one_patch_s, self.patch_fill, self.svg_text,
                                                           self.svg_txt_one_patch_e)

        return self.svg_txt_one_patch

    def run_no_PM(self):
        self.svg_text = simplify_patch_actionv_no_PM(svg_text=self.svg_text, width=self.width,
                                                            color=self.color, gt_patch=self.gt_patch,
                                                            patch_coord=self.patch_coord, coord_bais=self.coord_bais,
                                                            div_num=self.div_num, patch_num=self.patch_num)
        self.svg_txt_one_patch = '''{}\n{}\n{}{}'''.format(self.svg_txt_one_patch_s, self.patch_fill, self.svg_text,
                                                           self.svg_txt_one_patch_e)

        return self.svg_txt_one_patch


def print_num(num, content):
    for i in range(num):
        print("{}==== {}\n".format(content, i))
