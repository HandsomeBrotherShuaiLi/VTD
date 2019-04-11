import argparse,os

import numpy as np
from PIL import Image, ImageDraw
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from Detection.AdvancedEAST import cfg
from Detection.AdvancedEAST.label import point_inside_of_quad
from Detection.AdvancedEAST.network import East
from Detection.AdvancedEAST.preprocess import resize_image
from Detection.AdvancedEAST.nms import nms


def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))


def cut_text_line(geo, scale_ratio_w, scale_ratio_h, im_array, s,predict_img_dir,img_name):
    geo /= [scale_ratio_w, scale_ratio_h]
    p_min = np.amin(geo, axis=0)
    p_max = np.amax(geo, axis=0)
    min_xy = p_min.astype(int)
    max_xy = p_max.astype(int) + 2
    sub_im_arr = im_array[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0], :].copy()
    for m in range(min_xy[1], max_xy[1]):
        for n in range(min_xy[0], max_xy[0]):
            if not point_inside_of_quad(n, m, geo, p_min, p_max):
                sub_im_arr[m - min_xy[1], n - min_xy[0], :] = 255
    sub_im = image.array_to_img(sub_im_arr, scale=False)
    # sub_im.save(img_path + '_subim%d.jpg' % s)
    sub_im.save(os.path.join(predict_img_dir,img_name.replace('.jpg','_subim{}.jpg'.format(s))))


def predict(east_detect, img_path, pixel_threshold, quiet=True):
    img = image.load_img(img_path)
    d_wight, d_height = resize_image(img, cfg.max_predict_img_size)
    img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    img = image.img_to_array(img)
    img = preprocess_input(img, mode='tf')
    x = np.expand_dims(img, axis=0)
    y = east_detect.predict(x)

    y = np.squeeze(y, axis=0)
    y[:, :, :3] = sigmoid(y[:, :, :3])
    cond = np.greater_equal(y[:, :, 0], pixel_threshold)
    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms = nms(y, activation_pixels)
    with Image.open(img_path) as im:
        im_array = image.img_to_array(im.convert('RGB'))
        d_wight, d_height = resize_image(im, cfg.max_predict_img_size)
        scale_ratio_w = d_wight / im.width
        scale_ratio_h = d_height / im.height
        im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
        quad_im = im.copy()
        draw = ImageDraw.Draw(im)
        for i, j in zip(activation_pixels[0], activation_pixels[1]):
            px = (j + 0.5) * cfg.pixel_size
            py = (i + 0.5) * cfg.pixel_size
            line_width, line_color = 1, 'red'
            if y[i, j, 1] >= cfg.side_vertex_pixel_threshold:
                if y[i, j, 2] < cfg.trunc_threshold:
                    line_width, line_color = 2, 'yellow'
                elif y[i, j, 2] >= 1 - cfg.trunc_threshold:
                    line_width, line_color = 2, 'green'
            draw.line([(px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                       (px + 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                       (px + 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                       (px - 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                       (px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size)],
                      width=line_width, fill=line_color)
        im.save(img_path + '_act.jpg')
        quad_draw = ImageDraw.Draw(quad_im)
        txt_items = []
        for score, geo, s in zip(quad_scores, quad_after_nms,
                                 range(len(quad_scores))):
            if np.amin(score) > 0:
                quad_draw.line([tuple(geo[0]),
                                tuple(geo[1]),
                                tuple(geo[2]),
                                tuple(geo[3]),
                                tuple(geo[0])], width=2, fill='red')
                if cfg.predict_cut_text_line:
                    cut_text_line(geo, scale_ratio_w, scale_ratio_h, im_array,
                                  img_path, s)
                rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
                rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
                txt_item = ','.join(map(str, rescaled_geo_list))
                txt_items.append(txt_item + '\n')
            elif not quiet:
                print('quad invalid with vertex num less then 4.')
        quad_im.save(img_path + '_predict.jpg')
        if cfg.predict_write2txt and len(txt_items) > 0:
            with open(img_path[:-4] + '.txt', 'w') as f_txt:
                f_txt.writelines(txt_items)


def predict_txt(east_detect, img_path, txt_path, pixel_threshold, quiet=False):
    img = image.load_img(img_path)
    d_wight, d_height = resize_image(img, cfg.max_predict_img_size)
    scale_ratio_w = d_wight / img.width
    scale_ratio_h = d_height / img.height
    img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    img = image.img_to_array(img)
    img = preprocess_input(img, mode='tf')
    x = np.expand_dims(img, axis=0)
    y = east_detect.predict(x)

    y = np.squeeze(y, axis=0)
    y[:, :, :3] = sigmoid(y[:, :, :3])
    cond = np.greater_equal(y[:, :, 0], pixel_threshold)
    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms = nms(y, activation_pixels)

    txt_items = []
    for score, geo in zip(quad_scores, quad_after_nms):
        if np.amin(score) > 0:
            rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
            rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
            txt_item = ','.join(map(str, rescaled_geo_list))
            txt_items.append(txt_item + '\n')
        elif not quiet:
            print('quad invalid with vertex num less then 4.')
    if cfg.predict_write2txt and len(txt_items) > 0:
        with open(txt_path, 'w') as f_txt:
            f_txt.writelines(txt_items)

def predict_new(east_detect, img_name,img_path, pixel_threshold,predict_img_dir,predict_geo_dir, quiet=True):
    img = image.load_img(img_path)
    d_wight, d_height = resize_image(img, cfg.max_predict_img_size)
    img = img.resize((max(1,d_wight), max(d_height,1)), Image.ANTIALIAS)
    img = image.img_to_array(img)
    img = preprocess_input(img, mode='tf')
    x = np.expand_dims(img, axis=0)
    fail=[]
    try:
        y = east_detect.predict(x)
        y = np.squeeze(y, axis=0)
        y[:, :, :3] = sigmoid(y[:, :, :3])
        cond = np.greater_equal(y[:, :, 0], pixel_threshold)
        activation_pixels = np.where(cond)
        quad_scores, quad_after_nms = nms(y, activation_pixels)
        with Image.open(img_path) as im:
            # im是原始大小的图片,x是(0,resize后的图片)
            quad_im = im.copy()
            im_array = image.img_to_array(im)
            d_wight, d_height = resize_image(im, cfg.max_predict_img_size)
            scale_ratio_w = d_wight / im.width
            scale_ratio_h = d_height / im.height
            im = im.resize((d_wight, d_height), Image.ANTIALIAS)
            # im 也resize了

            # draw = ImageDraw.Draw(im)
            # for i, j in zip(activation_pixels[0], activation_pixels[1]):
            #     px = (j + 0.5) * cfg.pixel_size
            #     py = (i + 0.5) * cfg.pixel_size
            #     line_width, line_color = 1, 'red'
            #     if y[i, j, 1] >= cfg.side_vertex_pixel_threshold:
            #         if y[i, j, 2] < cfg.trunc_threshold:
            #             line_width, line_color = 2, 'yellow'
            #         elif y[i, j, 2] >= 1 - cfg.trunc_threshold:
            #             line_width, line_color = 2, 'green'
            #     draw.line([(px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
            #                (px + 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
            #                (px + 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
            #                (px - 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
            #                (px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size)],
            #               width=line_width, fill=line_color)
            #
            # # im.save(img_path + '_act.jpg')
            # im.save(os.path.join(predict_img_dir,img_name.replace('.jpg','_act.jpg')))
            # print(os.path.join(predict_img_dir,img_path.replace('.jpg','_act.jpg')))
            quad_draw = ImageDraw.Draw(quad_im)
            txt_items = []
            for score, geo, s in zip(quad_scores, quad_after_nms,
                                     range(len(quad_scores))):
                if np.amin(score) > 0:
                    rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
                    quad_draw.line([tuple(rescaled_geo[0]),
                                    tuple(rescaled_geo[1]),
                                    tuple(rescaled_geo[2]),
                                    tuple(rescaled_geo[3]),
                                    tuple(rescaled_geo[0])], width=2, fill='red')
                    if cfg.predict_cut_text_line:
                        cut_text_line(geo, scale_ratio_w, scale_ratio_h, im_array,
                                      s, predict_img_dir, img_name)

                    rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
                    txt_item = ','.join(map(str, rescaled_geo_list))
                    txt_items.append(txt_item + '\n')
                elif not quiet:
                    print('quad invalid with vertex num less then 4.')
            quad_im.save(os.path.join(predict_img_dir, img_name.replace('.jpg', '_predict.jpg')))
            if cfg.predict_write2txt and len(txt_items) > 0:
                with open(os.path.join(predict_geo_dir, img_name.replace('.jpg', '_geo.txt')), 'w',
                          encoding='utf-8') as f_txt:
                    f_txt.writelines(txt_items)

        print(img_path + '  DONE!!!')
    except Exception as e:
        print(e)
        print(img_path+' Failed')
        fail.append(img_path)





