import torch
import cv2
import numpy as np


def get_coord(width=128, device='cuda:0'):
    coord = torch.zeros([1, 2, width, width])
    for i in range(width):
        for j in range(width):
            coord[0, 0, i, j] = i / (width - 1.)
            coord[0, 1, i, j] = j / (width - 1.)
    coord = coord.to(device)  # Coordconv
    return coord


def normal(x, width):
    return (int)(x * (width - 1) + 0.5)


def gray_div_01_tensor(x):
    x = torch.round(x * 10) / 10
    return x


def Decoder_cv(f, width=128):
    # size = 10
    '''
    (x0; y0; x1; y1; x2; y2; r0; t0; r1; t1; R; G; B)
    (r0; t0), (r1; t1) control the thickness and transparency of the two endpoints of the curve, respectively.
    '''
    stroke = []
    for action in f:
        x0, y0, x1, y1, x2, y2, z0, z2 = action
        x1 = x0 + (x2 - x0) * x1
        y1 = y0 + (y2 - y0) * y1
        x0 = normal(x0, width * 2)
        x1 = normal(x1, width * 2)
        x2 = normal(x2, width * 2)
        y0 = normal(y0, width * 2)
        y1 = normal(y1, width * 2)
        y2 = normal(y2, width * 2)
        z0 = (int)(1 + z0 * width // 2)
        z2 = (int)(1 + z2 * width // 2)
        canvas = np.ones([width * 2, width * 2]).astype('float32') * 255
        rate = 1000
        tmp = 1. / rate
        w = 0
        for i in range(rate):
            t = i * tmp
            x = (int)((1 - t) * (1 - t) * x0 + 2 * t * (1 - t) * x1 + t * t * x2)
            y = (int)((1 - t) * (1 - t) * y0 + 2 * t * (1 - t) * y1 + t * t * y2)
            z = (int)((1 - t) * z0 + t * z2)
            cv2.circle(canvas, (x, y), radius=z, color=w, thickness=-1)  # -1 means filling
        result = cv2.resize(canvas, dsize=(width, width))
        result = result.astype('float32') / 255
        result = np.round(result)
        stroke.append(result)
    stroke = np.array(stroke).astype('float32')
    stroke = torch.from_numpy(stroke).cuda()
    return stroke


def decode(x, canvas):  # b * (10 + 3)
    x = x.view(-1, 9)
    f = x[:, :8]
    ac_or_not = x[:, -1:].round()
    color = x[:, -1:]
    color = gray_div_01_tensor(color)
    canvas = gray_div_01_tensor(canvas)
    # color = torch.from_numpy(np.around(np.array(x[:, -2:-1].detach().cpu()), 1)).cuda()

    # d = torch.round(test_render(f))

    d = torch.round(Decoder_cv(f))  # torch.Size([96, 8])
    stroke = 1 - d
    # s.save_middle_img(d, name='d')
    # s.save_middle_img(stroke, name='stroke0')
    stroke = stroke.view(-1, 128, 128, 1)

    # color_stroke = stroke * x[:, -1:].view(-1, 1, 1, 1)
    color_stroke = stroke * color.view(-1, 1, 1, 1)
    # s.save_middle_img(color_stroke, name='color_stroke')
    stroke = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    stroke = stroke.view(-1, 1, 1, 128, 128)
    color_stroke = color_stroke.view(-1, 1, 1, 128, 128)
    for i in range(1):
        # s.save_middle_img(canvas, name='c0')
        canvas = canvas * (1 - stroke[:, i])
        # s.save_middle_img(canvas, name='c1')
        # s.save_middle_img((stroke[:, i]), name='d1')
        canvas = canvas + color_stroke[:, i]
        # s.save_middle_img(canvas, name='c2'+str(ac_or_not[i].cpu()))
        # s.save_middle_img(color_stroke[:, i], name='color_stroke')
        # s.add_num()
    return canvas


def decode_list(x, canvas):  # b * (10 + 3)
    canvas = decode(x, canvas)
    res = []
    for i in range(1):
        res.append(canvas)
    return canvas, res


def small2large(x, divide, width=128):
    # (d * d, width, width) -> (d * width, d * width)
    x = x.reshape(divide, divide, width, width, -1)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(divide * width, divide * width, -1)
    return x


def large2small(x, canvas_cnt, divide, width=128):
    # (d * width, d * width) -> (d * d, width, width)
    x = x.reshape(divide, width, divide, width, 1)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(canvas_cnt, width, width, 1)
    return x


def smooth(img, divide, width):
    def smooth_pix(img, tx, ty):
        if tx == divide * width - 1 or ty == divide * width - 1 or tx == 0 or ty == 0:
            return img
        img[tx, ty] = (img[tx, ty] + img[tx + 1, ty] + img[tx, ty + 1] + img[tx - 1, ty] + img[tx, ty - 1] + img[
            tx + 1, ty - 1] + img[tx - 1, ty + 1] + img[tx - 1, ty - 1] + img[tx + 1, ty + 1]) / 9
        return img

    for p in range(divide):
        for q in range(divide):
            x = p * width
            y = q * width
            for k in range(width):
                img = smooth_pix(img, x + k, y + width - 1)
                if q != divide - 1:
                    img = smooth_pix(img, x + k, y + width)
            for k in range(width):
                img = smooth_pix(img, x + width - 1, y + k)
                if p != divide - 1:
                    img = smooth_pix(img, x + width, y + k)
    return img


def save_img(res, imgid, divide_number, width, origin_shape, divide=False):
    output = res.detach().cpu().numpy()  # d * d, 3, width, width
    output = np.transpose(output, (0, 2, 3, 1))
    if divide:
        output = small2large(output, divide_number, width)
        output = smooth(output, width=width, divide=divide_number)
    else:
        output = output[0]
    output = (output * 255).astype('uint8')
    output = cv2.resize(output, origin_shape)
    path = 'output/' + str(imgid) + '.png'
    cv2.imwrite(path, output)
    print(path)


def binarize(img):
    (h, w) = img.shape
    img = img.astype('float32') / 255
    img = np.around(img, 1) * 255
    img = img.astype('uint8')
    img = np.require(img, dtype='f4', requirements=['O', 'W'])
    # cat.flags.writeable = True
    for j in range(w):
        for i in range(h):
            pix = img[i, j]
            if pix >= 200:
                img[i, j] = 255
            if pix <= 50:
                img[i, j] = 0
    return img


def gray_div(img):
    img = img.astype('float32') / 255
    img = np.around(img, 1) * 255
    img = img.astype('uint8')
    return img


def img2patch(div_num, width=128):
    coord_x = np.ones([div_num, div_num])
    coord_y = np.ones([div_num, div_num])
    # x col, y row
    for i in range(div_num):
        for j in range(div_num):
            coord_y[i][j] = i * (width - 1)
            coord_x[i][j] = j * (width - 1)
    return [coord_x, coord_y]

def Gray_to_Hex(gray):
    RGB = [gray, gray, gray]
    color = '#'
    for i in RGB:
        num = int(i)
        color += str(hex(num))[-2:].replace('x', '0').upper()
    # print(color)
    return color