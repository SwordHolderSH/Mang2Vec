import os
import numpy as np
import cairosvg
import math
import mythread
from cairosvg import svg2png
import shutil

class Decode_np(object):
    def __init__(self, div_num, use_PM=True):
        self.np_path = './output_np/'
        self.div_num = div_num
        self.svg_txt_total = ''
        self.num = 0
        self.SAVE_PATH = './output/'
        self.all_patch_actions = np.load(self.np_path + 'all_patch_actions.npy')
        self.patch_done_list = np.load(self.np_path + 'patch_done_list.npy')
        self.config_list = np.load(self.np_path + 'config_list.npy', allow_pickle=True)
        self.fill_list = np.load(self.np_path + 'fill_list.npy', allow_pickle=True)
        self.patch_bgcolor_list = np.load(self.np_path + 'patch_bgcolor_list.npy', allow_pickle=True)
        self.START_TXT = str(np.load(self.np_path + 'START_TXT.npy', allow_pickle=True))
        self.END_TXT = str(np.load(self.np_path + 'END_TXT.npy', allow_pickle=True))
        self.div_num = np.load(self.np_path + 'div_num.npy', allow_pickle=True)
        self.width = np.load(self.np_path + 'width.npy', allow_pickle=True)
        self.output_width = self.width * self.div_num
        self.use_PM = use_PM

    def patchnum2coord(self, patch_num):
        i = patch_num % self.div_num
        j = math.floor(patch_num / self.div_num)
        return [i, j]

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

    def draw_decode(self):
        if self.use_PM:
            '''simplify actions'''
            for patch_num in range(len(self.all_patch_actions)):
                self.patch_done = self.patch_done_list[patch_num]
                config = self.config_list[patch_num]
                if self.patch_done == 'not_done':
                    m = mythread.myThread(config, self.patch_bgcolor_list, self.fill_list)
                    svg_txt_one_patch = m.run()
                    self.svg_txt_total = '''{}\n{}'''.format(self.svg_txt_total, svg_txt_one_patch)
                elif self.patch_done == 'not_done_fill':  # OK
                    m = mythread.myThread(config, self.patch_bgcolor_list, self.fill_list)
                    svg_txt_one_patch = m.run()
                    # svg_txt_one_patch = ""
                    self.svg_txt_total = '''{}\n{}'''.format(self.svg_txt_total, svg_txt_one_patch)
                elif self.patch_done == 'done_fill':  # OK
                    s = '''<g style="clip-path: url(#clipPath{}); ">'''.format(patch_num)
                    e = '''</g>'''
                    f = self.fill_list[patch_num]
                    svg_txt_one_patch = "{}\n{}\n{}".format(s, f, e)
                    self.svg_txt_total = '''{}\n{}'''.format(self.svg_txt_total, svg_txt_one_patch)
                elif self.patch_done == 'done':
                    pass
                self.save_results()
                self.num = self.num + 1
        else:
            for patch_num in range(len(self.all_patch_actions)):
                self.patch_done = self.patch_done_list[patch_num]
                config = self.config_list[patch_num]
                if self.patch_done == 'not_done':
                    m = mythread.myThread(config, self.patch_bgcolor_list, self.fill_list)
                    svg_txt_one_patch = m.run_no_PM()
                    self.svg_txt_total = '''{}\n{}'''.format(self.svg_txt_total, svg_txt_one_patch)
                elif self.patch_done == 'not_done_fill':  # OK
                    m = mythread.myThread(config, self.patch_bgcolor_list, self.fill_list)
                    svg_txt_one_patch = m.run_no_PM()
                    # svg_txt_one_patch = ""
                    self.svg_txt_total = '''{}\n{}'''.format(self.svg_txt_total, svg_txt_one_patch)
                elif self.patch_done == 'done_fill':  # OK
                    s = '''<g style="clip-path: url(#clipPath{}); ">'''.format(patch_num)
                    e = '''</g>'''
                    f = self.fill_list[patch_num]
                    svg_txt_one_patch = "{}\n{}\n{}".format(s, f, e)
                    # svg_txt_one_patch = a + config[0]
                    self.svg_txt_total = '''{}\n{}'''.format(self.svg_txt_total, svg_txt_one_patch)
                elif self.patch_done == 'done':
                    pass

                self.save_results()
                self.num = self.num + 1


def del_file(filepath):
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

def save_np(mat, name, path='./output_np/'):
    save_path = '{}{}.npy'.format(path, name)
    np.save(save_path, mat)