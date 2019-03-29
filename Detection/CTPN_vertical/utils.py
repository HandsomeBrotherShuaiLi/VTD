import os,xml.etree.ElementTree as ET
import numpy as np
from PIL import Image,ImageDraw
from torch.utils.data import Dataset
import torch
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


def generate_anchors(featuremap_size=(64,32),scale=16,kmeans=False,k=None):
    """

    :param featuremap_size: 对于vgg16 feature map是/16 /16 .对于其他是不一样的
    :param scale:
    :return:
    """
    if kmeans==False:
        heights=[2]*10
        widths=[1, 2, 4,6,8,12,16,26,32,50]

        heights=np.array(heights).reshape(len(heights),1)
        widths=np.array(widths).reshape(len(widths),1)

        base_anchor=np.array([0,0,1,1])
        xt=(base_anchor[0]+base_anchor[2])*0.5
        yt=(base_anchor[1]+base_anchor[3])*0.5
        x1=xt-widths*0.5
        y1=yt-heights*0.5
        x2=xt+widths*0.5
        y2=yt+heights*0.5
        base_anchor=np.hstack((x1,y1,x2,y2))

        h,w=featuremap_size
        shift_x=np.arange(0,w)*scale
        shift_y=np.arange(0,h)*scale
        anchor=[]
        for i in shift_y:
            for j in shift_x:
                anchor.append(base_anchor+[j,i,j,i])
        return np.array(anchor).reshape(-1,4)
    else:
        """
        使用kmeans聚类生成最合适的k个anchor,参考yolo_v3
        """
        pass
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

def cal_overlaps(boxes1,boxes2):
    """

    :param boxes1: box list anchors
    :param boxes2: box list ground truth box
    :return:
    """
    area1=(boxes1[:,0]-boxes1[:,2])*(boxes1[:,1]-boxes1[:,3])
    area2 = (boxes2[:, 0] - boxes2[:, 2]) * (boxes2[:, 1] - boxes2[:, 3])
    overlaps=np.zeros((boxes1.shape[0],boxes2.shape[0]))
    for i in range(boxes1.shape[0]):
        overlaps[i][:]=cal_iou(boxes1[i],area1[i],boxes2,area2)
    return overlaps

def cal_rpn(imgsize,featuresize,scale,gtboxes,iou_positive,iou_negative,rpn_positive_num,rpn_total_num):
    #使用PIL,img size 是（w,h)
    imgw,imgh=imgsize
    base_anchors=generate_anchors(featuresize,scale)
    overlaps=cal_overlaps(base_anchors,gtboxes)
    #0 represents negative box,1 represents positive box
    labels=np.empty(base_anchors.shape[0])
    labels.fill(-1)
    gt_argmax_overlaps=overlaps.argmax(axis=0)
    anchor_argmax_overlaps=overlaps.argmax(axis=1)
    anchor_max_overlas=overlaps[range(overlaps.shape[0]),anchor_argmax_overlaps]
    labels[anchor_max_overlas>iou_positive]=1
    labels[anchor_max_overlas<iou_negative]=0
    labels[gt_argmax_overlaps]=1

    outside_anchor=np.where((base_anchors[:,0]<0)|(base_anchors[:,1]<0)|
                            (base_anchors[:,2]>=imgw)|(base_anchors[:,3]>=imgh))[0]
    labels[outside_anchor]=-1

    fg_index=np.where(labels==1)[0]
    if len(fg_index)>rpn_positive_num:
        labels[np.random.choice(fg_index,len(fg_index)-rpn_positive_num,replace=False)]=-1
    bg_index=np.where(labels==0)[0]
    num_bg=rpn_total_num-np.sum(labels==1)
    if len(bg_index)>num_bg:
        labels[np.random.choice(bg_index,len(bg_index)-num_bg,replace=False)]=-1
    #calculate bbox targets
    bbox_targets=bbox_transform(base_anchors,gtboxes[anchor_argmax_overlaps,:])
    return [labels,bbox_targets],base_anchors

def bbox_transform(anchors,gtboxes):
    """
    compute the relative predicted horizonal(x) coordinates Vc,Vw
    becuase we have fixed height 16 or 32, width is variable
    :param anchors:
    :param gtboxes:
    :return:
    """
    regr=np.zeros((anchors.shape[0],2))
    Cx=(gtboxes[:,0]+gtboxes[:,2])*0.5
    # a means anchor
    Cxa=(anchors[:,0]+anchors[:,2])*0.5

    W=gtboxes[:,2]-gtboxes[:,0]+1.0
    Wa=anchors[:,2]-anchors[:,0]+1.0
    Vc=(Cx-Cxa)/Wa
    Vw=np.log(W/Wa)
    return np.vstack((Vc,Vw)).transpose()

def bbox_trasfor_inv(anchor,regr,scale):
    """
    anchor:(Nsample,4)
    regr=(Nsample,2)
    预测的时候，反向得到GT
    :param anchor:
    :param regr:
    :param scale:
    :return:
    """
    Cxa=(anchor[:,0]+anchor[:,2])*0.5 #anchor的中心点x坐标
    Wa=anchor[:,2]-anchor[:,0]+1 # anchor's width

    Delta_cx=regr[...,0] # 中心点x坐标偏移
    Delta_w=regr[...,1] # width's delta value

    GT_x=Delta_cx*Wa+Cxa # Ground Truth’s  x coordination
    GT_w=np.exp(Delta_w)*Wa# Ground truth's width

    Cya=(anchor[:,1]+anchor[:,3])*0.5 # anchor中心的y

    y1=Cya-scale*0.5
    x1=GT_x-GT_w*0.5
    y2=Cya+scale*0.5
    x2=GT_x+GT_w*0.5

    bbox=np.vstack((x1,y1,x2,y2)).transpose()
    return bbox

def filter_bbox(bbox, minsize):
    ws = bbox[:, 2] - bbox[:, 0] + 1
    hs = bbox[:, 3] - bbox[:, 1] + 1
    keep = np.where((ws >= minsize) & (hs >= minsize))[0]
    return keep

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


class MyDataSet(Dataset):
    def __init__(self,image_dir,label_dir,image_shape=None,base_model_name='vgg16',iou_positive=0.7,iou_negative=0.3,
                 iou_select=0.7,rpn_positive_num=150,rpn_total_num=300):
        """

        :param image_dir:
        :param label_dir: xml dir
        :param image_shape:
        """
        if os.path.isdir(image_dir)==False:
            raise Exception('{} not exist'.format(image_dir))
        if os.path.isdir(label_dir)==False:
            raise Exception('{} not exist'.format(label_dir))
        self.image_dir=image_dir
        self.label_dir=label_dir
        if len(os.listdir(self.image_dir))!=len(os.listdir(self.label_dir)):
            raise Exception('image number != label number ')
        else:
            print('image number={}, label number={}'.format(len(os.listdir(self.image_dir)),len(os.listdir(self.label_dir))))
        self.image_shape=image_shape
        self.image_list=os.listdir(self.image_dir)
        self.label_list=os.listdir(self.label_dir)
        self.base_model_name=base_model_name
        self.iou_positive=iou_positive
        self.iou_negative=iou_negative
        self.iou_select=iou_select
        self.rpn_total_num=rpn_total_num
        self.rpn_positive_num=rpn_positive_num

    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self,index):
        image_name=self.image_list[index]
        image_path=os.path.join(self.image_dir,image_name)
        label_path=os.path.join(self.label_dir,image_name.replace('.jpg','.xml'))
        gtboxes,xml_filename=readxml(label_path)
        if xml_filename!=image_name:
            raise Exception('read xml error')
        img=Image.open(image_path)
        """
        修改原始图像后，对应的box也要修改
        TODO!
        """
        if self.image_shape!=None:
            #image_shape=(width,height)
            original_size=img.size
            x_scale = self.image_shape[0] / original_size[0]
            y_scale = self.image_shape[1] / original_size[1]
            newbox = []
            for i in range(len(gtboxes)):
                newbox.append(
                    [gtboxes[i][0] * x_scale, gtboxes[i][1] * y_scale, gtboxes[i][2] * x_scale, gtboxes[i][3] * y_scale]
                )
            img=img.resize((self.image_shape[0],self.image_shape[1]),Image.ANTIALIAS)
            gtboxes=np.array(newbox)
        w,h=img.size
        if self.base_model_name=='vgg16':
            scale=16
        else:
            scale=32
        [cls,regr],_=cal_rpn(imgsize=(w,h),featuresize=(int(h/scale),int(w/scale)),scale=scale,gtboxes=gtboxes,
                             iou_positive=self.iou_positive,iou_negative=self.iou_negative,
                             rpn_total_num=self.rpn_total_num,rpn_positive_num=self.rpn_positive_num)
        img=np.array(img)
        img=(img / 255.0) * 2.0 - 1.0
        regr=np.hstack([cls.reshape(cls.shape[0],1),regr])
        cls=np.expand_dims(cls,axis=0)
        #pytorch 是channel first的,所以得把numpy 图片的第三维channel放在首位
        img=torch.from_numpy(img.transpose([2,0,1])).float()
        cls=torch.from_numpy(cls).float()
        regr=torch.from_numpy(regr).float()
        return img.cpu(),cls.cpu(),regr.cpu()

class TextLineCfg:
    SCALE=600
    MAX_SCALE=1200
    TEXT_PROPOSALS_WIDTH=2
    MIN_NUM_PROPOSALS = 2
    MIN_RATIO=0.5
    LINE_MIN_SCORE=0.9
    MAX_HORIZONTAL_GAP=60
    TEXT_PROPOSALS_MIN_SCORE=0.7
    TEXT_PROPOSALS_NMS_THRESH=0.3
    MIN_V_OVERLAPS=0.6
    MIN_SIZE_SIM=0.6
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
        # print(precursors)
        if precursors==[]:
            return False
        if self.scores[index] >= np.max(self.scores[precursors]):
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