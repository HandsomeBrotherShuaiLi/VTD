import numpy as np
import os,xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
from glob import glob
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
from PIL import Image,ImageDraw
from math import *
anchor_scale = 16
#
IOU_NEGATIVE = 0.3
IOU_POSITIVE = 0.7
IOU_SELECT = 0.7

RPN_POSITIVE_NUM = 150
RPN_TOTAL_NUM = 300

# bgr  can find from  here https://github.com/fchollet/deep-learning-models/blob/master/imagenet_utils.py
IMAGE_MEAN = [123.68, 116.779, 103.939]

DEBUG = True
def rotate(img_path):

    img=cv2.imread(img_path)
    height, width = img.shape[:2]

    degree = 90
    # 旋转后的尺寸
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))  # 这个公式参考之前内容
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    matRotation[0, 2] += (widthNew - width) / 2  # 因为旋转之后,坐标系原点是新图像的左上角,所以需要根据原图做转化
    matRotation[1, 2] += (heightNew - height) / 2

    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))

    return Image.fromarray(imgRotation)

def drawRect(gtboxes,img):
    draw=ImageDraw.Draw(img)
    for i in gtboxes:
        draw.rectangle((i[0],i[1],i[2],i[3]),outline='red')
    img.show()

def readxml(path):
    gtboxes=[]
    tree=ET.parse(path)
    root=tree.getroot()
    image_name = tree.find('filename').text
    for obj in root.iter('object'):
        try:
            bnd=obj.find('bndbox')
            xmin=int(bnd.find('xmin').text)
            ymin=int(bnd.find('ymin').text)
            xmax=int(bnd.find('xmax').text)
            ymax=int(bnd.find('ymax').text)
            gtboxes.append((xmin,ymin,xmax,ymax))
        except Exception as e:
            raise Exception(e+" when {} loading xml file, there are errors".format(path))
    return np.array(gtboxes),image_name


def gen_anchor(featuresize, scale):
    """
    gen base anchor from feature map [HXW][10][4]
    reshape  [HXW][10][4] to [HXWX10][4]
    生成的锚框是相对于原图的，即原图中每16像素就有10个锚框
    """

    heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    widths = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16]

    # gen k=9 anchor size (h,w)
    heights = np.array(heights).reshape(len(heights), 1)
    widths = np.array(widths).reshape(len(widths), 1)

    # 锚框大小为16像素
    base_anchor = np.array([0, 0, 15, 15])
    # center x,y
    xt = (base_anchor[0] + base_anchor[2]) * 0.5
    yt = (base_anchor[1] + base_anchor[3]) * 0.5

    # x1 y1 x2 y2
    x1 = xt - widths * 0.5
    y1 = yt - heights * 0.5
    x2 = xt + widths * 0.5
    y2 = yt + heights * 0.5

    # 一组十个锚框
    base_anchor = np.hstack((x1, y1, x2, y2))

    h, w = featuresize
    shift_x = np.arange(0, w) * scale
    shift_y = np.arange(0, h) * scale
    # apply shift
    anchor = []
    for i in shift_y:
        for j in shift_x:
            anchor.append(base_anchor + [j, i, j, i])
    return np.array(anchor).reshape((-1, 4))


def cal_iou(box1, box1_area, boxes2, boxes2_area):
    """
    box1 [x1,y1,x2,y2]
    boxes2 [Msample,x1,y1,x2,y2]
    """
    x1 = np.maximum(box1[0], boxes2[:, 0])
    x2 = np.minimum(box1[2], boxes2[:, 2])
    y1 = np.maximum(box1[1], boxes2[:, 1])
    y2 = np.minimum(box1[3], boxes2[:, 3])

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    iou = intersection / (box1_area + boxes2_area[:] - intersection[:])
    return iou


def cal_overlaps(boxes1, boxes2):
    """
    boxes1 [Nsample,x1,y1,x2,y2]  anchor
    boxes2 [Msample,x1,y1,x2,y2]  grouth-box

    """
    area1 = (boxes1[:, 0] - boxes1[:, 2]) * (boxes1[:, 1] - boxes1[:, 3])  # (Nsample, 1)
    area2 = (boxes2[:, 0] - boxes2[:, 2]) * (boxes2[:, 1] - boxes2[:, 3])  # (Msample, 1)

    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))  # (Nsample, Msample)

    # calculate the intersection of  boxes1(anchor) and boxes2(GT box)
    for i in range(boxes1.shape[0]):
        overlaps[i][:] = cal_iou(boxes1[i], area1[i], boxes2, area2)

    return overlaps


def bbox_transfrom(anchors, gtboxes):
    """
    anchors: (Nsample, 4)
    gtboxes: (Nsample, 4)
     compute relative predicted vertical coordinates Vc ,Vh
        with respect to the bounding box location of an anchor
    """
    Cy = (gtboxes[:, 1] + gtboxes[:, 3]) * 0.5  # (Nsample, )
    Cya = (anchors[:, 1] + anchors[:, 3]) * 0.5  # (Nsample, )
    h = gtboxes[:, 3] - gtboxes[:, 1] + 1.0  # (Nsample, )
    ha = anchors[:, 3] - anchors[:, 1] + 1.0  # (Nsample, )

    Vc = (Cy - Cya) / ha  # (Nsample, )
    Vh = np.log(h / ha)  # (Nsample, )

    ret = np.vstack((Vc, Vh))

    return ret.transpose()  # (Nsample, 2)


def bbox_transfor_inv(anchor, regr):
    """
    anchor: (NSample, 4)
    regr: (NSample, 2)
    根据锚框和偏移量反向得到GTBox
    """

    Cya = (anchor[:, 1] + anchor[:, 3]) * 0.5  # 锚框y中心点
    ha = anchor[:, 3] - anchor[:, 1] + 1

    Vcx = regr[..., 0]  # y中心点偏移
    Vhx = regr[..., 1]  # 高度偏移

    Cyx = Vcx * ha + Cya  # GTBox y中心点
    hx = np.exp(Vhx) * ha  # GTBox 高
    xt = (anchor[:, 0] + anchor[:, 2]) * 0.5  # 锚框x中心点

    x1 = xt - 16 * 0.5
    y1 = Cyx - hx * 0.5
    x2 = xt + 16 * 0.5
    y2 = Cyx + hx * 0.5
    bbox = np.vstack((x1, y1, x2, y2)).transpose()

    return bbox


def clip_box(bbox, im_shape):
    # x1 >= 0
    bbox[:, 0] = np.maximum(np.minimum(bbox[:, 0], im_shape[1] - 1), 0)
    # y1 >= 0
    bbox[:, 1] = np.maximum(np.minimum(bbox[:, 1], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    bbox[:, 2] = np.maximum(np.minimum(bbox[:, 2], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    bbox[:, 3] = np.maximum(np.minimum(bbox[:, 3], im_shape[0] - 1), 0)

    return bbox


def filter_bbox(bbox, minsize):
    ws = bbox[:, 2] - bbox[:, 0] + 1
    hs = bbox[:, 3] - bbox[:, 1] + 1
    keep = np.where((ws >= minsize) & (hs >= minsize))[0]
    return keep


def cal_rpn(imgsize, featuresize, scale, gtboxes):
    """
    gtboxes: (Msample, 4)
    """
    imgh, imgw = imgsize

    # gen base anchor
    base_anchor = gen_anchor(featuresize, scale)  # (Nsample, 4)

    # calculate iou
    overlaps = cal_overlaps(base_anchor, gtboxes)  # (Nsample, Msample)

    # init labels -1 don't care  0 is negative  1 is positive
    labels = np.empty(base_anchor.shape[0])
    labels.fill(-1)  # (Nsample,)

    # for each GT box corresponds to an anchor which has highest IOU
    gt_argmax_overlaps = overlaps.argmax(axis=0)  # (Msample, )

    # the anchor with the highest IOU overlap with a GT box
    anchor_argmax_overlaps = overlaps.argmax(axis=1)  # (Nsample, )
    anchor_max_overlaps = overlaps[range(overlaps.shape[0]), anchor_argmax_overlaps]  # (Nsample, )

    # IOU > IOU_POSITIVE
    labels[anchor_max_overlaps > IOU_POSITIVE] = 1
    # IOU <IOU_NEGATIVE
    labels[anchor_max_overlaps < IOU_NEGATIVE] = 0
    # ensure that every GT box has at least one positive RPN region
    labels[gt_argmax_overlaps] = 1

    # only keep anchors inside the image
    outside_anchor = np.where(
        (base_anchor[:, 0] < 0) |
        (base_anchor[:, 1] < 0) |
        (base_anchor[:, 2] >= imgw) |
        (base_anchor[:, 3] >= imgh)
    )[0]
    labels[outside_anchor] = -1

    # 剔除掉多余的正负样例
    # subsample positive labels ,if greater than RPN_POSITIVE_NUM(default 128)
    fg_index = np.where(labels == 1)[0]
    if (len(fg_index) > RPN_POSITIVE_NUM):
        labels[np.random.choice(fg_index, len(fg_index) - RPN_POSITIVE_NUM, replace=False)] = -1

    # subsample negative labels
    bg_index = np.where(labels == 0)[0]
    num_bg = RPN_TOTAL_NUM - np.sum(labels == 1)
    if (len(bg_index) > num_bg):
        # print('bgindex:',len(bg_index),'num_bg',num_bg)
        labels[np.random.choice(bg_index, len(bg_index) - num_bg, replace=False)] = -1

    # calculate bbox targets
    # debug here
    bbox_targets = bbox_transfrom(base_anchor, gtboxes[anchor_argmax_overlaps, :])
    # bbox_targets=[]

    return [labels, bbox_targets], base_anchor


def get_session(gpu_fraction=0.6):
    '''''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads, allow_soft_placement=True))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))


class random_uniform_num():
    """
    uniform random
    """

    def __init__(self, total, start=0):
        self.total = total
        self.range = [i for i in range(total)]
        np.random.shuffle(self.range)
        self.index = start

    def get(self, batch_size):
        ret = []
        if self.index + batch_size > self.total:
            piece1 = self.range[self.index:]
            np.random.shuffle(self.range)
            self.index = (self.index + batch_size) - self.total
            piece2 = self.range[0:self.index]
            ret.extend(piece1)
            ret.extend(piece2)
        else:
            ret = self.range[self.index:self.index + batch_size]
            self.index = self.index + batch_size
        return ret


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # Sort from high to low

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def gen_sample(xmlpath, imgpath, batchsize=1):
    """
    由于图像大小不定，批处理大小只能为1
    """

    # list xml
    xmlfiles = glob(xmlpath + '/*.xml')
    rd = random_uniform_num(len(xmlfiles))
    xmlfiles = np.array(xmlfiles)

    while True:
        shuf = xmlfiles[rd.get(1)]
        gtbox, imgfile = readxml(shuf[0])
        img = cv2.imread(imgpath + "\\" + imgfile)
        h, w, c = img.shape

        # clip image
        if np.random.randint(0, 100) > 50:
            img = img[:, ::-1, :]
            newx1 = w - gtbox[:, 2] - 1
            newx2 = w - gtbox[:, 0] - 1
            gtbox[:, 0] = newx1
            gtbox[:, 2] = newx2

        [cls, regr], _ = cal_rpn((h, w), (int(h / 16), int(w / 16)), 16, gtbox)
        # zero-center by mean pixel
        m_img = img - IMAGE_MEAN
        m_img = np.expand_dims(m_img, axis=0)

        regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])

        #
        cls = np.expand_dims(cls, axis=0)
        cls = np.expand_dims(cls, axis=1)
        # regr = np.expand_dims(regr,axis=1)
        regr = np.expand_dims(regr, axis=0)

        yield m_img, {'rpn_class_reshape': cls, 'rpn_regress_reshape': regr}


def rpn_test():
    xmlpath = 'G:\data\VOCdevkit\VOC2007\Annotations\img_4375.xml'
    imgpath = 'G:\data\VOCdevkit\VOC2007\JPEGImages\img_4375.jpg'
    gtbox, _ = readxml(xmlpath)
    img = cv2.imread(imgpath)
    h, w, c = img.shape
    [cls, regr], base_anchor = cal_rpn((h, w), (int(h / 16), int(w / 16)), 16, gtbox)
    print(cls.shape)
    print(regr.shape)

    regr = np.expand_dims(regr, axis=0)
    inv_anchor = bbox_transfor_inv(base_anchor, regr)
    anchors = inv_anchor[cls == 1]
    anchors = anchors.astype(int)
    for i in anchors:
        cv2.rectangle(img, (i[0], i[1]), (i[2], i[3]), (255, 0, 0), 3)
    plt.imshow(img)


def threshold(coords, min_, max_):
    return np.maximum(np.minimum(coords, max_), min_)

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    boxes[:, 0::2]=threshold(boxes[:, 0::2], 0, im_shape[1]-1)
    boxes[:, 1::2]=threshold(boxes[:, 1::2], 0, im_shape[0]-1)
    return boxes


class Graph:
    def __init__(self, graph):
        self.graph=graph

    def sub_graphs_connected(self):
        sub_graphs=[]
        for index in range(self.graph.shape[0]):
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v=index
                sub_graphs.append([v])
                while self.graph[v, :].any():
                    v=np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)
        return sub_graphs
class TextLineCfg:
    SCALE=600
    MAX_SCALE=1200
    TEXT_PROPOSALS_WIDTH=16
    MIN_NUM_PROPOSALS = 2
    MIN_RATIO=0.5
    LINE_MIN_SCORE=0.9
    MAX_HORIZONTAL_GAP=60
    TEXT_PROPOSALS_MIN_SCORE=0.7
    TEXT_PROPOSALS_NMS_THRESH=0.3
    MIN_V_OVERLAPS=0.6
    MIN_SIZE_SIM=0.6

class TextProposalGraphBuilder:
    """
        Build Text proposals into a graph.
    """
    def get_successions(self, index):
            box=self.text_proposals[index]
            results=[]
            for left in range(int(box[0])+1, min(int(box[0])+TextLineCfg.MAX_HORIZONTAL_GAP+1, self.im_size[1])):
                adj_box_indices=self.boxes_table[left]
                for adj_box_index in adj_box_indices:
                    if self.meet_v_iou(adj_box_index, index):
                        results.append(adj_box_index)
                if len(results)!=0:
                    return results
            return results

    def get_precursors(self, index):
        box=self.text_proposals[index]
        results=[]
        for left in range(int(box[0])-1, max(int(box[0]-TextLineCfg.MAX_HORIZONTAL_GAP), 0)-1, -1):
            adj_box_indices=self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results)!=0:
                return results
        return results

    def is_succession_node(self, index, succession_index):
        precursors=self.get_precursors(succession_index)
        if self.scores[index]>=np.max(self.scores[precursors]):
            return True
        return False

    def meet_v_iou(self, index1, index2):
        def overlaps_v(index1, index2):
            h1=self.heights[index1]
            h2=self.heights[index2]
            y0=max(self.text_proposals[index2][1], self.text_proposals[index1][1])
            y1=min(self.text_proposals[index2][3], self.text_proposals[index1][3])
            return max(0, y1-y0+1)/min(h1, h2)

        def size_similarity(index1, index2):
            h1=self.heights[index1]
            h2=self.heights[index2]
            return min(h1, h2)/max(h1, h2)

        return overlaps_v(index1, index2)>=TextLineCfg.MIN_V_OVERLAPS and \
               size_similarity(index1, index2)>=TextLineCfg.MIN_SIZE_SIM

    def build_graph(self, text_proposals, scores, im_size):
        self.text_proposals=text_proposals
        self.scores=scores
        self.im_size=im_size
        self.heights=text_proposals[:, 3]-text_proposals[:, 1]+1

        boxes_table=[[] for _ in range(self.im_size[1])]
        for index, box in enumerate(text_proposals):
            boxes_table[int(box[0])].append(index)
        self.boxes_table=boxes_table

        graph=np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)

        for index, box in enumerate(text_proposals):
            successions=self.get_successions(index)
            if len(successions)==0:
                continue
            succession_index=successions[np.argmax(scores[successions])]
            if self.is_succession_node(index, succession_index):
                # NOTE: a box can have multiple successions(precursors) if multiple successions(precursors)
                # have equal scores.
                graph[index, succession_index]=True
        return Graph(graph)
class TextProposalConnector:
    def __init__(self):
        self.graph_builder=TextProposalGraphBuilder()

    def group_text_proposals(self, text_proposals, scores, im_size):
        graph=self.graph_builder.build_graph(text_proposals, scores, im_size)
        return graph.sub_graphs_connected()

    def fit_y(self, X, Y, x1, x2):
        len(X)!=0
        # if X only include one point, the function will get line y=Y[0]
        if np.sum(X==X[0])==len(X):
            return Y[0], Y[0]
        p=np.poly1d(np.polyfit(X, Y, 1))
        return p(x1), p(x2)

    def get_text_lines(self, text_proposals, scores, im_size):
        # tp=text proposal
        tp_groups=self.group_text_proposals(text_proposals, scores, im_size)
        text_lines=np.zeros((len(tp_groups), 5), np.float32)

        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes=text_proposals[list(tp_indices)]

            x0=np.min(text_line_boxes[:, 0])
            x1=np.max(text_line_boxes[:, 2])

            offset=(text_line_boxes[0, 2]-text_line_boxes[0, 0])*0.5

            lt_y, rt_y=self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0+offset, x1-offset)
            lb_y, rb_y=self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0+offset, x1-offset)

            # the score of a text line is the average score of the scores
            # of all text proposals contained in the text line
            score=scores[list(tp_indices)].sum()/float(len(tp_indices))

            text_lines[index, 0]=x0
            text_lines[index, 1]=min(lt_y, rt_y)
            text_lines[index, 2]=x1
            text_lines[index, 3]=max(lb_y, rb_y)
            text_lines[index, 4]=score

        text_lines=clip_boxes(text_lines, im_size)

        text_recs = np.zeros((len(text_lines), 9), np.float)
        index = 0
        for line in text_lines:
            xmin,ymin,xmax,ymax=line[0],line[1],line[2],line[3]
            text_recs[index, 0] = xmin
            text_recs[index, 1] = ymin
            text_recs[index, 2] = xmax
            text_recs[index, 3] = ymin
            text_recs[index, 4] = xmin
            text_recs[index, 5] = ymax
            text_recs[index, 6] = xmax
            text_recs[index, 7] = ymax
            text_recs[index, 8] = line[4]
            index = index + 1

        return text_recs


class TextProposalConnectorOriented:
    """
        Connect text proposals into text lines
    """

    def __init__(self):
        self.graph_builder = TextProposalGraphBuilder()

    def group_text_proposals(self, text_proposals, scores, im_size):
        graph = self.graph_builder.build_graph(text_proposals, scores, im_size)
        return graph.sub_graphs_connected()

    def fit_y(self, X, Y, x1, x2):
        len(X) != 0
        # if X only include one point, the function will get line y=Y[0]
        if np.sum(X == X[0]) == len(X):
            return Y[0], Y[0]
        p = np.poly1d(np.polyfit(X, Y, 1))
        return p(x1), p(x2)

    def get_text_lines(self, text_proposals, scores, im_size):
        """
        text_proposals:boxes

        """
        # tp=text proposal
        tp_groups = self.group_text_proposals(text_proposals, scores, im_size)  # 首先还是建图，获取到文本行由哪几个小框构成

        text_lines = np.zeros((len(tp_groups), 8), np.float32)

        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes = text_proposals[list(tp_indices)]  # 每个文本行的全部小框
            X = (text_line_boxes[:, 0] + text_line_boxes[:, 2]) / 2  # 求每一个小框的中心x，y坐标
            Y = (text_line_boxes[:, 1] + text_line_boxes[:, 3]) / 2

            z1 = np.polyfit(X, Y, 1)  # 多项式拟合，根据之前求的中心店拟合一条直线（最小二乘）

            x0 = np.min(text_line_boxes[:, 0])  # 文本行x坐标最小值
            x1 = np.max(text_line_boxes[:, 2])  # 文本行x坐标最大值

            offset = (text_line_boxes[0, 2] - text_line_boxes[0, 0]) * 0.5  # 小框宽度的一半

            # 以全部小框的左上角这个点去拟合一条直线，然后计算一下文本行x坐标的极左极右对应的y坐标
            lt_y, rt_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0 + offset, x1 - offset)
            # 以全部小框的左下角这个点去拟合一条直线，然后计算一下文本行x坐标的极左极右对应的y坐标
            lb_y, rb_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0 + offset, x1 - offset)

            score = scores[list(tp_indices)].sum() / float(len(tp_indices))  # 求全部小框得分的均值作为文本行的均值

            text_lines[index, 0] = x0
            text_lines[index, 1] = min(lt_y, rt_y)  # 文本行上端 线段 的y坐标的小值
            text_lines[index, 2] = x1
            text_lines[index, 3] = max(lb_y, rb_y)  # 文本行下端 线段 的y坐标的大值
            text_lines[index, 4] = score  # 文本行得分
            text_lines[index, 5] = z1[0]  # 根据中心点拟合的直线的k，b
            text_lines[index, 6] = z1[1]
            height = np.mean((text_line_boxes[:, 3] - text_line_boxes[:, 1]))  # 小框平均高度
            text_lines[index, 7] = height + 2.5

        text_recs = np.zeros((len(text_lines), 9), np.float)
        index = 0
        for line in text_lines:
            b1 = line[6] - line[7] / 2  # 根据高度和文本行中心线，求取文本行上下两条线的b值
            b2 = line[6] + line[7] / 2
            x1 = line[0]
            y1 = line[5] * line[0] + b1  # 左上
            x2 = line[2]
            y2 = line[5] * line[2] + b1  # 右上
            x3 = line[0]
            y3 = line[5] * line[0] + b2  # 左下
            x4 = line[2]
            y4 = line[5] * line[2] + b2  # 右下
            disX = x2 - x1
            disY = y2 - y1
            width = np.sqrt(disX * disX + disY * disY)  # 文本行宽度

            fTmp0 = y3 - y1  # 文本行高度
            fTmp1 = fTmp0 * disY / width
            x = np.fabs(fTmp1 * disX / width)  # 做补偿
            y = np.fabs(fTmp1 * disY / width)
            if line[5] < 0:
                x1 -= x
                y1 += y
                x4 += x
                y4 -= y
            else:
                x2 += x
                y2 += y
                x3 -= x
                y3 -= y
            text_recs[index, 0] = x1
            text_recs[index, 1] = y1
            text_recs[index, 2] = x2
            text_recs[index, 3] = y2
            text_recs[index, 4] = x3
            text_recs[index, 5] = y3
            text_recs[index, 6] = x4
            text_recs[index, 7] = y4
            text_recs[index, 8] = line[4]
            index = index + 1

        return text_recs

if __name__=='__main__':
    xmlpath = 'D:\py_projects\data_new\data_new\data\\annotation\\img_calligraphy_00001_bg.xml'
    imgpath = 'D:\py_projects\data_new\data_new\data\\train_img\\img_calligraphy_00001_bg.jpg'
    gtboxes, _ = readxml(xmlpath)
    img = rotate(imgpath)
    original_size = img.size
    x_scale = 256 / original_size[0]
    y_scale = 512 / original_size[1]
    newbox = []
    for i in range(len(gtboxes)):
        newbox.append(
            [gtboxes[i][0] * x_scale, gtboxes[i][1] * y_scale, gtboxes[i][2] * x_scale,
             gtboxes[i][3] * y_scale]
        )
    img = img.resize((256, 512), Image.ANTIALIAS)
    gtboxes = np.array(newbox)
    w, h = img.size
    [cls, regr], base_anchor = cal_rpn((w, h), (h // 16, w // 16), 16, gtboxes, iou_positive=0.7, iou_negative=0.3,
                                       rpn_total_num=300, rpn_positive_num=150)
    print(cls.shape, regr.shape)
    regr = np.expand_dims(regr, axis=0)
    inv_anchor = bbox_trasfor_inv(base_anchor, regr, scale=16)
    anchors = inv_anchor[cls == 0]
    anchors = anchors.astype(int)
    for i in anchors:
        print(i)
    drawRect(anchors, img)