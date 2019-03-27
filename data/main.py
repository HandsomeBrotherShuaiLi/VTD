import cv2,numpy as np
import pandas as pd
import threadpool,os
import xml.etree.ElementTree as ET
from xml.dom.minidom import Document

class Data(object):
    def __init__(self,chinese_dict_path,original_concat_csv_path,anno_path,
                 train_img_for_detection_path,train_img_for_recognition_path,train_img_labels_path,
                 train_img_for_recognition_output_path=None,
                 train_img_detection_output_path=None,test_img_detection_output_path=None,
                 test_img_recognition_output_path=None):
        """

        :param chinese_dict_path: 输入的中文字典路径
        :param original_concat_csv_path: 联合csv文件路径，包含所有location以及class信息
        :param train_img_for_detection_path: 针对定位的训练图片路径
        :param train_img_for_recognition_path: 针对识别的Path
        :param train_img_labels_path: label path
        :param test_img_detection_output_path: 输出的路径
        :param train_img_for_recognition_output_path:
        :param train_img_detection_output_path:
        """
        self.chinese=open(chinese_dict_path,'r',encoding='utf-8').readlines()[0].split(',')
        self.chinese_dict={c:i for i,c in enumerate(self.chinese)}
        self.original_concat_csv_path=original_concat_csv_path
        self.train_img_for_detection_path=train_img_for_detection_path
        self.train_img_for_recognition_path=train_img_for_recognition_path
        self.train_img_labels_path=train_img_labels_path
        self.train_img_for_recognition_output_path=train_img_for_recognition_output_path
        self.train_img_detection_output_path=train_img_detection_output_path
        self.test_img_detection_output_path=test_img_detection_output_path
        self.test_img_recognition_path=test_img_recognition_output_path
        self.anno_path=anno_path
    def csv2xml_single_thread(self,img_path):
        """

        :param img_path: 来自完整的图片文件夹的路径文件名
        :return:
        """
        global lack_img
        img_folder_path = self.train_img_for_detection_path
        csv_path = self.original_concat_csv_path
        converted_xml_path = self.anno_path

        if (img_path.endswith('.png') or img_path.endswith('.jpg')) and img_path != '1.png':
            doc = Document()
            annotation = doc.createElement('annotation')
            doc.appendChild(annotation)
            folder = doc.createElement('folder')
            annotation.appendChild(folder)
            folder_content = doc.createTextNode(img_folder_path)
            folder.appendChild(folder_content)

            filename = doc.createElement('filename')
            annotation.appendChild(filename)
            filename_content = doc.createTextNode(img_path)
            filename.appendChild(filename_content)

            """
            read img file 
            """
            try:
                img = cv2.imread(os.path.join(img_folder_path, img_path))
            except Exception as e:
                lack_img.append(str(e)+' '+img_path)
                return
            h, w, c = img.shape
            size = doc.createElement('size')
            annotation.appendChild(size)
            width = doc.createElement('width')
            size.appendChild(width)
            width_content = doc.createTextNode(str(w))
            width.appendChild(width_content)

            height = doc.createElement('height')
            size.appendChild(height)
            height_content = doc.createTextNode(str(h))
            height.appendChild(height_content)

            channel = doc.createElement('depth')
            size.appendChild(channel)
            channel_txt = doc.createTextNode(str(c))
            channel.appendChild(channel_txt)

            df = pd.read_csv(csv_path)
            res = df[df.FileName == img_path]
            # print(res)
            for i in res.index:
                # print('i=',i)
                object_new = doc.createElement('object')
                annotation.appendChild(object_new)
                # name = doc.createElement('name')
                # object_new.appendChild(name)
                # name_txt = doc.createTextNode(res.loc[i, 'text'])
                # name.appendChild(name_txt)

                # for c in res.loc[i, 'text']:
                #     chinese.add(c)
                bndbox = doc.createElement('bndbox')
                object_new.appendChild(bndbox)
                """
                因为rpn是基于xmin xmax等四个坐标来做的，所以不得不把坐标位置修改成四元组
                所以这只能基于长方形训练，对于普通四边形不可行
                """
                x = np.array([res.loc[i, 'x1'], res.loc[i, 'x2'], res.loc[i, 'x3'], res.loc[i, 'x4']])
                y = np.array([res.loc[i, 'y1'], res.loc[i, 'y2'], res.loc[i, 'y3'], res.loc[i, 'y4']])

                xmin_int = x.min()
                xmax_int = x.max()
                ymin_int = y.min()
                ymax_int = y.max()

                xmin = doc.createElement('xmin')
                bndbox.appendChild(xmin)
                xmin_text = doc.createTextNode(str(xmin_int))
                xmin.appendChild(xmin_text)

                ymin = doc.createElement('ymin')
                bndbox.appendChild(ymin)
                ymin_text = doc.createTextNode(str(ymin_int))
                ymin.appendChild(ymin_text)

                xmax = doc.createElement('xmax')
                bndbox.appendChild(xmax)
                xmax_text = doc.createTextNode(str(xmax_int))
                xmax.appendChild(xmax_text)

                ymax = doc.createElement('ymax')
                bndbox.appendChild(ymax)
                ymax_text = doc.createTextNode(str(ymax_int))
                ymax.appendChild(ymax_text)

            xml_path = os.path.join(converted_xml_path, img_path.strip('.jpg') + 'g.xml')
            with open(xml_path, 'w', encoding='utf-8') as f:
                doc.writexml(f, indent='\t', encoding='utf-8', newl='\n', addindent='\t')
                print(xml_path + ' done!')

    def csv2xml_multi_thread(self):
        """
        把CSV文件->xml
        :return:
        """
        global lack_img
        lack_img=[]
        threadcount = 1280
        pool = threadpool.ThreadPool(threadcount)
        request = threadpool.makeRequests(self.csv2xml_single_thread,
                                          os.listdir(self.train_img_for_detection_path))
        [pool.putRequest(req) for req in request]
        pool.wait()
        with open('../../info/lack_img.txt','w',encoding='utf-8') as f:
            f.write('\n'.join(lack_img))
        print(lack_img)

    def test_stage_one(self):
        annolist=os.listdir(self.anno_path)
        imglist=os.listdir(self.train_img_for_detection_path)
        csv_img_names=set(list(pd.read_csv(self.original_concat_csv_path)['FileName']))
        print(len(csv_img_names),len(annolist),len(imglist))
        for i in csv_img_names:
            if i not in imglist:
                print(i+'不在imglist')
            if i.replace('.jpg','.xml') not in annolist:
                print(i.replace('.jpg','.xml')+'不存在')
        print('Well Done!')

    def cut_image(self):
        """
        检查是不是切割完整，以及生成的label文件是不是完整的，如果不是，那么生成新的文件。
        :return:
        """
        labels_file=open(self.train_img_labels_path,'r',encoding='utf-8').readlines()
        labels_name_list=[i.split(' ')[0].strip('\n') for i in labels_file]
        print(len(labels_name_list))
        csv_file=pd.read_csv(self.original_concat_csv_path)
        print(len(csv_file))
        cut_image_list=os.listdir(self.train_img_for_recognition_path)
        print(len(cut_image_list))
        original_img=os.listdir(self.train_img_for_detection_path)
        print(len(original_img))
        lack_cut_img=[]
        for img in original_img:
            res=csv_file[csv_file.FileName==img]
            index=list(res.index)
            for i in range(len(index)):
                newname=img.replace('.jpg','_'+str(i)+'.jpg')
                if newname in cut_image_list and newname in labels_name_list:
                    pass
                else:
                    print(newname,newname in cut_image_list,newname in labels_name_list)
                    x = np.array(
                        [res.loc[index[i], 'x1'], res.loc[index[i], 'x2'],
                         res.loc[index[i], 'x3'],
                         res.loc[index[i], 'x4']])
                    y = np.array(
                        [res.loc[index[i], 'y1'], res.loc[index[i], 'y2'],
                         res.loc[index[i], 'y3'],
                         res.loc[index[i], 'y4']])
                    xmin_int = x.min()
                    xmax_int = x.max()
                    ymin_int = y.min()
                    ymax_int = y.max()
                    if newname not in cut_image_list:
                        img_file=cv2.imread(os.path.join(self.train_img_for_detection_path,img))
                        img_temp = img_file[ymin_int:ymax_int, xmin_int:xmax_int, :]
                        cv2.imwrite(os.path.join(self.train_img_for_recognition_path,newname),img_temp)
                        print(newname+' has been cut down')
                    if newname not in labels_name_list:
                        with open(self.train_img_labels_path,'a',encoding='utf-8') as f:
                            temp=newname+' '.join([str(self.chinese_dict[i]) for i in list(str(res.loc[index[i],'text']).strip('\n'))])
                            f.write(temp)
                            print(temp)
                            print(temp,' write done!')

        labels_file = open(self.train_img_labels_path, 'r', encoding='utf-8').readlines()
        labels_name_list = [i.split(' ')[0].strip('\n') for i in labels_file]
        print(len(labels_name_list))
        csv_file = pd.read_csv(self.original_concat_csv_path)
        print(len(csv_file))
        cut_image_list = os.listdir(self.train_img_for_recognition_path)
        print(len(cut_image_list))
        original_img = os.listdir(self.train_img_for_detection_path)
        print(len(original_img))




