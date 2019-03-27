class ctpn(object):
    def __init__(self,base_model_name,scale,iou_positive=0.7,iou_negative=0.3,
                 iou_select=0.7,rpn_positive_num=150,rpn_total_num=300):
        self.base_model_name=base_model_name
        self.scale=scale
        self.iou_positive=iou_positive
        self.iou_negative=iou_negative
        self.iou_select=iou_select
        self.rpn_total_num=rpn_total_num
        self.rpn_positive_num=rpn_positive_num

