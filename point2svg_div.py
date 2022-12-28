# ! encoding:UTF-8
import cv2
import numpy as np
import os
from cairosvg import svg2png
import math
import fitCurves
import cairosvg
from PIL import Image
import torch.nn as nn
import time
import decode as d
import vectorize_utils as vu

def get_most_gray(img_patch, patch_size):
    # print(np.bincount(img_patch))
    img_f = img_patch.astype('uint8').flat
    img_patch = np.bincount(img_f)
    most_value = np.argmax(img_patch)
    most_count = np.max(img_patch)
    p = most_count / patch_size / patch_size
    # most_value = max(set(img_patch), key=img_patch.count)
    # if p >= 0.9 and (most_value >= 240 or most_value <= 20):
    return most_value, p


def patch_fill(img, div_num):
    (h, w) = img.shape
    max_size = max(h, w)
    patch_size = 0
    while True:
        if div_num * patch_size >= max_size:
            break
        else:
            patch_size = patch_size + 1

    size = div_num * patch_size
    img = cv2.resize(img, (size, size))
    img = vu.gray_div(img)
    patch_done_list = []
    patch_color_list = []
    # n = 0
    for r in range(div_num):
        for c in range(div_num):
            # a = Image.fromarray(img)
            # a.save('./output/total{}.png'.format(n))
            # n = n + 1
            gray, p = get_most_gray(
                img_patch=img[r * (patch_size): (r + 1) * (patch_size), c * (patch_size): (c + 1) * (patch_size)],
                patch_size=patch_size)
            if (gray != 255) and p >= 0.7:
                img[r * (patch_size): (r + 1) * (patch_size), c * (patch_size): (c + 1) * (patch_size)] = gray
                patch_color_list.append(gray)
            else:
                img[r * (patch_size): (r + 1) * (patch_size), c * (patch_size): (c + 1) * (patch_size)] = 255
                patch_color_list.append(gray)
            if (gray == 255) and p == 1:
                patch_done_list.append('done')
            elif (gray == 0) and p == 1:
                patch_done_list.append('done_fill')
            elif (gray != 255) and p >= 0.7:
                patch_done_list.append('not_done_fill')
            else:
                patch_done_list.append('not_done')

    I = Image.fromarray(img)
    I.save('./output/total.png')
    return img, patch_done_list


class Point2svg(object):
    def __init__(self, width, div_num, img_w, img_h, img, save_path='./output/', white_bg=True, init_num=0,
                 use_patch_fill=False, patch_done_list=[]):
        self.SAVE_PATH = save_path
        self.width = width
        self.div_num = div_num
        self.output_width = self.width * self.div_num
        self.coord_bais = vu.img2patch(div_num=self.div_num)
        self.clip_text = self.get_clip()
        self.scalex, self.scaley = self.get_scalexy(img_w, img_h)
        self.use_patch_fill = use_patch_fill
        self.patch_done_list = patch_done_list
        self.patch_svg_list_total = []
        self.all_patch_actions = []
        self.patch_bgcolor_list = []
        for l in range(self.div_num * self.div_num): self.all_patch_actions.append([]), self.patch_bgcolor_list.append(
            [])
        self.START_TXT = '''<svg xmlns="http://www.w3.org/2000/svg" version="1.1" height="{}" width="{}" viewBox="0 0 {} {}"><g transform="scale({}, {})">\n<rect x="0" y="0" width="{}" height="{}" fill="white"/>\n{}\n'''.format(
            self.output_width, self.output_width * self.scaley, self.output_width, self.output_width * self.scaley,
            self.scalex, self.scaley, self.output_width, self.output_width, self.clip_text)

        if self.use_patch_fill == True:
            self.patch_fill_svg(img=img, div_num=self.div_num, svg_size=self.output_width)
            # svg_div = myutil.get_svg_div(self.patch_fill_svg(img=img, div_num=self.div_num, svg_size=self.output_width))
        else:
            svg_div = '''\n'''
        self.element_list = []
        self.END_TXT = '''</g></svg>'''
        self.svg_txt_total = ""
        self.num = init_num
        self.layer_num = 0
        self.mseloss = nn.MSELoss()
        self.f = 0

        self.test_count = 0

    def img2patch(self):
        coord_x = np.ones([self.div_num, self.div_num])
        coord_y = np.ones([self.div_num, self.div_num])
        # x col, y row
        for i in range(self.div_num):
            for j in range(self.div_num):
                coord_y[i][j] = i * (self.width - 1)
                coord_x[i][j] = j * (self.width - 1)
        return [coord_x, coord_y]

    def get_clip(self):
        clip_text = ""
        for patch_num in range(self.div_num * self.div_num):
            patch_coord = self.patchnum2coord(patch_num=patch_num)
            [i, j] = patch_coord
            [cx, cy] = self.coord_bais
            baisx = cx[i][j]
            baisy = cy[i][j]
            patch_text = '''<defs><clippath id="clipPath{}"> <rect x="{}" y="{}" width="{}" height="{}"></rect></clippath></defs>'''.format(
                patch_num, baisy, baisx, self.width, self.width)
            clip_text = '''{}\n{}'''.format(clip_text, patch_text)
        return clip_text

    def patchnum2coord(self, patch_num):
        i = patch_num % self.div_num
        j = math.floor(patch_num / self.div_num)
        return [i, j]

    def patch_fill_svg(self, img, div_num, svg_size):
        self.done_fill_list = []
        self.not_done_fill_list = []
        self.fill_list = []
        for l in range(self.div_num * self.div_num): self.done_fill_list.append([]), self.not_done_fill_list.append(
            []), self.fill_list.append([])

        (h, w) = img.shape
        max_size = max(h, w)
        svg_patch_size = np.float(svg_size) / div_num

        patch_size = 0
        while True:
            if div_num * patch_size >= max_size:
                break
            else:
                patch_size = patch_size + 1

        size = div_num * patch_size
        img = cv2.resize(img, (size, size))
        img = vu.gray_div(img)
        svg_total = []
        l = div_num * div_num
        for patch_num in range(l):
            patch_coord = self.patchnum2coord(patch_num=patch_num)
            # self.patch_pic_reset(patch_coord=patch_coord)
            [c, r] = patch_coord
            yr = r * (svg_patch_size - 1)
            xr = c * (svg_patch_size - 1)
            gray, p = get_most_gray(
                img_patch=img[r * patch_size: (r + 1) * patch_size, c * patch_size: (c + 1) * patch_size],
                patch_size=patch_size)
            color = vu.Gray_to_Hex(gray)
            if gray != 255 and p >= 0.7 and p != 1:
                svg_txt = '''<rect x="{}" y="{}" width="{}" height="{}" fill="{}"/>'''.format(xr, yr,
                                                                                              svg_patch_size,
                                                                                              svg_patch_size,
                                                                                              color)
                svg_total.append(svg_txt)
                self.patch_bgcolor_list[patch_num].append(color)
                self.not_done_fill_list[patch_num].append(svg_txt)
                self.fill_list[patch_num].append(svg_txt)
            elif gray != 255 and p == 1:
                svg_txt = '''<rect x="{}" y="{}" width="{}" height="{}" fill="{}"/>'''.format(xr, yr,
                                                                                              svg_patch_size,
                                                                                              svg_patch_size,
                                                                                              color)
                self.done_fill_list[patch_num].append(svg_txt)
                self.patch_bgcolor_list[patch_num].append(color)
                self.fill_list[patch_num].append(svg_txt)
            else:
                self.done_fill_list[patch_num].append("")
                self.patch_bgcolor_list[patch_num].append(vu.Gray_to_Hex(255))
                self.fill_list[patch_num].append("")

    def get_scalexy(self, width0, height0):
        scale_y = height0 / width0
        scale_y = np.round(scale_y, 2)
        scale_x = 1
        return scale_x, scale_y

    def reset_gt_patch(self, gt):
        self.gt = np.array(gt.squeeze(1).cpu())

    def add_action_div(self, new_action, stroke_dim=8, color_dim=1):  # torch.Size([256, 65])
        new_action = new_action.detach().view(self.div_num * self.div_num, -1, stroke_dim + color_dim)
        new_action = new_action.permute(1, 0, 2)
        for a in range(new_action.shape[0]):  # 5
            one_action_in_all_patch = new_action[a]  # (256, 13)
            for patch_num in range(one_action_in_all_patch.shape[0]):  # div*div
                # patch_coord = self.patchnum2coord(patch_num=patch_action)
                action_single = one_action_in_all_patch[patch_num]
                self.add_to_listv2(action=action_single, patch_num=patch_num)

    def add_to_listv2(self, action, patch_num):
        action = np.array(action.cpu())
        self.all_patch_actions[patch_num].append(action)

    def normal(self, x, width):
        x = (width - 1) * x
        x = (int)(x + 0.5)
        return x

    def get_xyzk_by_t(self, f, t):
        [x0, y0, x1, y1, x2, y2, z0, z2] = f
        x = ((1 - t) * (1 - t) * x0 + 2 * t * (1 - t) * x1 + t * t * x2)
        y = ((1 - t) * (1 - t) * y0 + 2 * t * (1 - t) * y1 + t * t * y2)
        z = ((1 - t) * z0 + t * z2)
        k = self.get_bezier_k(f=f, t=t)
        return x, y, z, k

    def get_bezier_k(self, f, t):
        [x0, y0, x1, y1, x2, y2, z0, z2] = f
        yd = 2 * ((1 - t) * (y1 - y0) + t * (y2 - y1))
        xd = 2 * ((1 - t) * (x1 - x0) + t * (x2 - x1))
        if xd == 0:
            xd = 0.001
        if yd == 0:
            yd = 0.001
        k = yd / xd
        return k

    def get_tangent_point(self, xc, yc, r, k):
        # b = yc - (k * xc)
        if k == 0:
            k = 1e-8
        k2 = -1 / k
        # b2 = yc - (k2 * xc)
        dgree = np.arctan(k2)
        y_dis = np.sin(dgree) * r
        x_dis = np.cos(dgree) * r

        point1 = [xc + x_dis, yc + y_dis]
        point2 = [xc - x_dis, yc - y_dis]
        return point1, point2

    def get_get_tangent_point_by_t(self, p, t):
        xm, ym, zm, km = self.get_xyzk_by_t(f=p, t=t)
        if km == 0:
            km = 1e-8
        point0, point1 = self.get_tangent_point(xc=xm, yc=ym, r=zm, k=km)
        return point0, point1

    def get_cyclev2(self, x, y, r, color):
        svg_txt = '''<circle cx="{}" cy="{}" r="{}" fill="{}"/> \n'''.format(x, y, r, color)
        return svg_txt

    def curvepoint2svg(self, fit):
        svg_txt = ''' '''
        for curve in range(len(fit)):
            cure_point = fit[curve]
            x1, y1, x2, y2, ex, ey = cure_point[1][0], cure_point[1][1], cure_point[2][0], cure_point[2][1], \
                                     cure_point[3][0], cure_point[3][1]
            svg_txt = svg_txt + '''C {} {} {} {} {} {} '''.format(x1, y1, x2, y2, ex, ey)
        return svg_txt

    def get_edgev2(self, a0, f0, svg_edge_first, svg_edge_second, color):
        svg_edge = '''<path d =  "M {} {} {} L {} {} {} Z"  fill="{}" stroke="{}" stroke-width="{}" /> \n'''.format(
            str(a0[0]), str(a0[1]), svg_edge_first, str(f0[0]), str(f0[1]), svg_edge_second, color, 'none', '0')
        return svg_edge

    def mysvg2pdf(self, svg_txt, save_path):
        svg_txt = self.START_TXT + svg_txt + self.END_TXT
        exportPath = save_path
        exportFileHandle = open(exportPath, 'w')
        cairosvg.svg2pdf(bytestring=svg_txt, write_to=exportPath)
        exportFileHandle.close()

    def mysvg2png(self, svg_txt, save_path):
        svg_txt = self.START_TXT + svg_txt + self.END_TXT
        svg2png(bytestring=svg_txt, write_to=save_path, output_width=self.output_width,
                output_height=self.output_width)

    def get_svg_txt_by_path(self, action, width, patch_coord):
        width = width / 2
        action = np.array(action).astype('float32')
        point = action[:8]
        gray = action[-1:]
        gray = np.around(gray, 1)
        rgb = np.uint8(gray * 255)
        color = vu.Gray_to_Hex(rgb)
        # x0, y0, x1, y1, x2, y2, z0, z2 = point
        y0, x0, y1, x1, y2, x2, z0, z2 = point
        [i, j] = patch_coord
        [cx, cy] = self.coord_bais
        baisx = cx[i][j]
        baisy = cy[i][j]
        x1 = x0 + (x2 - x0) * x1
        y1 = y0 + (y2 - y0) * y1
        x0 = self.normal(x0, width * 2)
        x1 = self.normal(x1, width * 2)
        x2 = self.normal(x2, width * 2)
        y0 = self.normal(y0, width * 2)
        y1 = self.normal(y1, width * 2)
        y2 = self.normal(y2, width * 2)
        z0 = (1 + z0 * width // 2)
        z2 = (1 + z2 * width // 2)

        if x0 == x2:
            x2 = x2 + 1e-8
        if y0 == y2:
            y2 = y2 + 1e-8

        p = [x0 + baisx, y0 + baisy, x1 + baisx, y1 + baisy, x2 + baisx, y2 + baisy, z0, z2]
        [y0, x0, y1, x1, y2, x2, z0, z2] = p
        p = [x0, y0, x1, y1, x2, y2, z0, z2]
        # [y0, x0, y1, x1, y2, x2, z0, z2] = p

        rate = 1e8
        tmp = 1. / rate

        '''用来找路径的点'''
        t_range = np.linspace(0, 1, 1000)
        t_range = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1]
        t_point = []
        for i in t_range:
            t_point.append(i * rate * tmp)

        list_point1 = []
        list_point2 = []
        for i in range(len(t_point)):
            points1, points2 = self.get_get_tangent_point_by_t(p=p, t=t_point[i])
            list_point1.append(points1)
            list_point2.append(points2)

        edge_point1_list = np.array(list_point1)
        list_point2.reverse()
        edge_point2_list = np.array(list_point2)

        '''根据点拟合曲线'''
        fitCubic = fitCurves.fitCurve(points=edge_point1_list, maxError=10)
        fitCubic2 = fitCurves.fitCurve(points=edge_point2_list, maxError=10)

        fit = np.round(fitCubic, 2)
        fit2 = np.round(fitCubic2, 2)

        '''开始和结束的圆'''
        start_cycle = self.get_cyclev2(x0, y0, z0, color)
        rate_end = 10000
        tmp_end = 1. / rate_end
        t_end = (rate_end - 1) * tmp_end
        endx, endy, endr = ((1 - t_end) * (1 - t_end) * x0 + 2 * t_end * (1 - t_end) * x1 + t_end * t_end * x2), (
                (1 - t_end) * (1 - t_end) * y0 + 2 * t_end * (1 - t_end) * y1 + t_end * t_end * y2), (
                                   (1 - t_end) * z0 + t_end * z2)
        end_cycle = self.get_cyclev2(np.around(endx, 2), np.around(endy, 2), np.around(endr, 2), color)
        '''边缘'''
        svg_edge_first = self.curvepoint2svg(fit=fit)
        svg_edge_second = self.curvepoint2svg(fit=fit2)
        # svg_edge = self.get_edgev2(np.round(a0, 1), np.round(f0, 1), svg_edge_first, svg_edge_second, color=color)
        svg_edge = self.get_edgev2(np.round(list_point1[0], 2), np.round(list_point2[0], 2), svg_edge_first,
                                   svg_edge_second, color=color)
        svg_txt_total = svg_edge + start_cycle + end_cycle
        # t_or_f = self.activate_action_path(para=[svg_txt_total, patch_coord])
        # if t_or_f is True or t_or_f is False:

        self.patch_svg_list.append(svg_txt_total)
        # self.patch_svg_list.append(start_cycle)
        # self.patch_svg_list.append(end_cycle)

        # return svg_txt_total

    def create_str_to_svg(self, svg_path, str_data):
        str_data = self.START_TXT + str_data + self.END_TXT
        # str_data = '''{}\n{}\n'''.format(str_data, self.END_TXT)
        if not os.path.exists(svg_path):
            with open(svg_path, "w") as f:
                f.write(str_data)
        else:
            os.remove(svg_path)
            with open(svg_path, "w") as f:
                f.write(str_data)

    def save_results(self):

        if self.num % 100 == 0 or self.num == len(self.config_list) - 1:
            print('num{}'.format(self.num))
            print('leng{}'.format(len(self.config_list) - 1))
            SAVE_PATH_SVG = self.SAVE_PATH + str(self.num) + '.svg'
            SAVE_PATH_PNG = self.SAVE_PATH + str(self.num) + '_svg.png'
            SAVE_PATH_PDF = self.SAVE_PATH + str(self.num) + '_svg.pdf'
            # self.create_str_to_svg(svg_path=SAVE_PATH_SVG, str_data=self.svg_txt_total)
            self.mysvg2png(svg_txt=self.svg_txt_total, save_path=SAVE_PATH_PNG)
            self.mysvg2pdf(svg_txt=self.svg_txt_total, save_path=SAVE_PATH_PDF)
            # self.mysvg2pdf(svg_txt=self.svg_txt_total, save_path=SAVE_PATH_SVG)
            self.create_str_to_svg(svg_path=SAVE_PATH_SVG, str_data=self.svg_txt_total)
            print(SAVE_PATH_PNG)


    def draw_action_list_for_all_patch(self, path_or_circle='circle'):
        self.config_list = []
        for patch_num in range(len(self.all_patch_actions)):
            one_patch_action = self.all_patch_actions[patch_num]
            patch_coord = self.patchnum2coord(patch_num=patch_num)
            '''reset each patch'''
            self.patch_done = self.patch_done_list[patch_num]
            if self.patch_done == 'not_done':
                self.patch_svg_list = []
                for action_single in one_patch_action:
                        self.get_svg_txt_by_path(action=action_single, width=self.width,
                                                   patch_coord=patch_coord)
                config = [self.patch_svg_list.copy(), self.width, self.gt[patch_num], patch_coord,
                          self.coord_bais, self.div_num, patch_num]
                self.config_list.append(config)
            elif self.patch_done == 'not_done_fill':
                self.patch_svg_list = []
                patch_coord = self.patchnum2coord(patch_num=patch_num)
                # a = self.not_done_fill_list[patch_num][0]
                # self.patch_svg_list.append(a)
                for action_single in one_patch_action:
                    self.get_svg_txt_by_path(action=action_single, width=self.width,
                                               patch_coord=patch_coord)
                config = [self.patch_svg_list.copy(), self.width, self.gt[patch_num], patch_coord,
                          self.coord_bais, self.div_num, patch_num]
                self.config_list.append(config)
            elif self.patch_done == 'done_fill':
                self.patch_svg_list = []
                patch_coord = self.patchnum2coord(patch_num=patch_num)
                for action_single in one_patch_action:
                    self.get_svg_txt_by_path(action=action_single, width=self.width,
                                               patch_coord=patch_coord)
                config = [self.patch_svg_list.copy(), self.width, self.gt[patch_num], patch_coord,
                          self.coord_bais, self.div_num, patch_num]
                self.config_list.append(config)
            elif self.patch_done == 'done':
                self.patch_svg_list = []
                patch_coord = self.patchnum2coord(patch_num=patch_num)
                self.patch_svg_list.append("")
                config = [self.patch_svg_list.copy(), self.width, self.gt[patch_num], patch_coord,
                          self.coord_bais, self.div_num, patch_num]
                self.config_list.append(config)

        s = time.time()
        temp_path = "./output_np/"
        d.del_file(temp_path)
        d.save_np(mat=np.array(self.all_patch_actions), name='all_patch_actions')
        d.save_np(mat=np.array(self.patch_done_list), name='patch_done_list')
        d.save_np(mat=np.array(self.config_list), name='config_list')
        d.save_np(mat=np.array(self.patch_bgcolor_list), name='patch_bgcolor_list')
        d.save_np(mat=np.array(self.fill_list), name='fill_list')
        d.save_np(mat=np.array(self.START_TXT), name='START_TXT')
        d.save_np(mat=np.array(self.END_TXT), name='END_TXT')
        d.save_np(mat=np.array(self.div_num), name='div_num')
        d.save_np(mat=np.array(self.width), name='width')
        e = time.time()
        print("Successfully saved temp variables to "+ temp_path)
        return e-s

