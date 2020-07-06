from typing import Any, Callable, Dict, Generator, List, Mapping, Union
import torch
from torch import nn
from functools import partial
from catalyst.contrib import registry
from catalyst.core import MetricCallback
from catalyst.core.registry import Callback, Criterion, Sampler
from torch.utils.data import DistributedSampler


@Callback
class IouCallbackSafe(MetricCallback):
    """Dice metric callback."""

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "iou",
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = None,
        **metric_kwargs
    ):
        """
        Args:
            input_key (str): input key to use for dice calculation;
                specifies our `y_true`
            output_key (str): output key to use for dice calculation;
                specifies our `y_pred`
        """
        super().__init__(
            prefix=prefix,
            metric_fn=iou_safe,
            input_key=input_key,
            output_key=output_key,
            eps=eps,
            threshold=threshold,
            activation=activation,
            **metric_kwargs
        )


def iou_safe(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    # values are discarded, only None check
    # used for compatibility with MultiMetricCallback
    classes: int = None,
    eps: float = 1e-7,
    threshold: float = None,
    activation: str = None, #"Softmax2d",
    ignore_index: int = 255,
    is_per_class: bool = True,
) -> Union[float, List[float]]:
    """
    Args:
        outputs (torch.Tensor): A list of predicted elements, multichannel
        targets (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold (float): threshold for outputs binarization
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ["none", "Sigmoid", "Softmax2d"]

    Returns:
        Union[float, List[float]]: IoU (Jaccard) score(s)
    """
    activation_fn = get_activation_fn(activation)
    outputs = activation_fn(outputs)

    if not is_per_class:
        # print('1hotIOU')
        if threshold is not None:
            outputs = (outputs > threshold).float()

        # ! fix backward compatibility
        if classes is not None:
            # if classes are specified we reduce across all dims except channels
            _sum = partial(torch.sum, dim=[0, 2, 3]) #[0, 2, 3]
        else:
            _sum = torch.sum

        if len(outputs.shape) > len(targets.shape):
            targets = torch.stack([targets == i for i in range(outputs.shape[1])], dim = 1)
        
        targets, outputs = targets.float(), outputs.float()
        # outputs[targets == ignore_index] = ignore_index

        intersection = _sum(targets * outputs)
        target_sum, output_sum = _sum(targets), _sum(outputs) 
        union = target_sum + output_sum - intersection
        # print('bin-tg', target_sum)
        # print('bin-op', output_sum)
        # print('bin-it', intersection)
        # print('bin-un', union)
        # this looks a bit awkward but `eps * (union == 0)` termq
        # makes sure that if I and U are both 0, than IoU == 1
        # and if U != 0 and I == 0 the eps term in numerator is zeroed out
        # i.e. (0 + eps) / (U - 0 + eps) doesn't happen
        iou_class = (intersection + eps * (union == 0).float()) / (union + eps)
        # print('bin is', intersection)
        # print('bin un', union)
        # print('bin iou', iou_class)
        # print('whole check dim', iou_class.shape)
        iou = torch.mean(iou_class)
        # print('\nIOU_whole',  iou_class.detach().cpu().numpy(), iou.detach().cpu().numpy())

    else:
        # print('histIOU')
        outputs = outputs.max(1)[1] # item 0 is the largest value along the comparing dimension, item 1 is the index of those largest values
        if len(targets.shape) > len(outputs.shape):
            targets = targets.max(1)[1]
        targets, outputs = targets.float(), outputs.float()
        area_intersection, area_union, area_target = intersectionAndUnionGPU(outputs, targets, classes, ignore_index=ignore_index)
        iou_class = (area_intersection + (area_union == 0) * eps ) / (area_union.float() + eps)
        # print('class check dim', iou_class.shape)
        # print('hist intersect', area_intersection)
        # print('hist union', area_union)
        # print('hist iou', iou_class)
        iou = torch.mean(iou_class)
        # print('\nIOU_class',  iou_class.cpu().numpy(), iou.cpu().numpy())

    return iou




def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def dice_by_sample(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
    threshold: float = None,
    activation: str = "Sigmoid",
):
    """Computes the dice metric.

    Args:
        outputs (list):  a list of predicted elements
        targets (list): a list of elements that are to be predicted
        eps (float): epsilon
        threshold (float): threshold for outputs binarization
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ["none", "Sigmoid", "Softmax2d"]

    Returns:
        double:  Dice score
    """
    activation_fn = get_activation_fn(activation)
    outputs = activation_fn(outputs)
    if threshold is not None:
        outputs = (outputs > threshold).float()
    dice = 0
    bs = outputs.shape[0]
    for i in range(bs):
        targets_i, outputs_i = targets[i,...], outputs[i, ...]
        intersection = torch.sum(targets_i * outputs_i)
        union = torch.sum(targets_i) + torch.sum(outputs_i)
        # this looks a bit awkward but `eps * (union == 0)` term
        # makes sure that if I and U are both 0, than Dice == 1
        # and if U != 0 and I == 0 the eps term in numerator is zeroed out
        # i.e. (0 + eps) / (U - 0 + eps) doesn't happen
        dice += (2 * intersection + eps * (union == 0)) / (union + eps)

    return dice /bs 



def get_activation_fn(activation: str = None):
    """Returns the activation function from ``torch.nn`` by its name."""
    if activation is None or activation.lower() == "none":
        activation_fn = lambda x: x  # noqa: E731
    else:
        activation_fn = torch.nn.__dict__[activation]()
    return activation_fn


# def tversky_loss(y_true, y_pred, alpha=0.5,
#                  beta=0.8, weight=(0.25, 1, 1, 1.5)):  # , 3, 3
#     """

#     :param y_true:
#     :param y_pred:
#     :param y_pred:
#     :param alpha: # 待修改项，调节假阳，默认为0.5
#     :param beta: # 待修改项，调节假阴，默认为0.5
#     :param weight: # 待修改项， 人为设定权重
#     :return:
#     """
#     class_n = K.get_variable_shape(y_pred)[-1]  # 待修改项，总分类数
#     print('number of class %d' % class_n)
#     total_loss = 0.
#     for i in range(class_n):
#         temp_true = y_true[..., i]  # G
#         temp_pred = y_pred[..., i]  # P
#         TP = K.sum(temp_true * temp_pred)  # G∩P，真阳
#         FN = K.sum(temp_true) - K.sum(temp_true * temp_pred)  # G-(G∩P),假阴
#         FP = K.sum(temp_pred) - K.sum(temp_true * temp_pred)  # P-(G∩P),假阳
#         temp_loss = 1 - (TP + 1e-10) / (TP + alpha * FN + beta * FP + 1e-10)
#         if weight is not None:
#             temp_loss *= weight[i]
#         total_loss += temp_loss
#     weight_sum = 1 if weight is None else sum(weight)
#     tversky_loss = total_loss / weight_sum
#     return tversky_loss


# # 基于交叉熵的focal_loss
# def ce_focal_loss(y_true, y_pred, gamma=2):
#     '''
#     结合Cross_Entropy的focal_loss
#     计算公式：(1-pt)**γ*(-y_true*log(pt))
#     '''
#     # 可人为设定，默认为2
#     cross_entropy = -y_true * K.log(y_pred + 1e-10)
#     weight = K.pow((1 - y_pred), gamma)
#     loss = weight * cross_entropy
#     ce_loss = K.sum(loss) / K.sum(y_true)
#     return ce_loss



# # loss function for segmentation models

# def top_k_loss(y_true, y_pred, t = 0.5):
#     """
#     search for hard pixels within the current mini-batch to calculate loss
#     hard pixels defined as those with the prediction probability of correct class less than threshold
#     Simply, drop pixels when they are too easy for the model
#     in practice, increase threshold t for mini-batch with good performance
#     decrease t for those with bad performance


#     originally proposed by Wu et al in 2016 in http://arxiv.org/abs/1605.06885
#     later adopted by Deepmind in segmenting Hneck OARs http://arxiv.org/abs/1809.04430
#     :param y_true:
#     :param y_pred:
#     :param t: threshold of prediction probability to drop a pixel when calculating loss
#     :return:
#     """
#     # y_true = np.array(y_true, dtype= np.float32)
#     # y_pred = np.array(y_pred, dtype= np.float32)
#     smooth = 1.0
#     target_class_pred = y_true[...,1:] * y_pred[...,1:]
#     hard_pixels_0 = tf.greater(target_class_pred, tf.zeros_like(target_class_pred))
#     hard_pixels_t = tf.less_equal(target_class_pred, tf.zeros_like(target_class_pred) + tf.constant(t))
#     hard_pixels_bool = tf.logical_and(hard_pixels_0, hard_pixels_t)

#     hard_pixels_pred_log = tf.log(tf.boolean_mask(y_pred[...,1:], hard_pixels_bool))

#     # print(np.argmax(y_true, axis=-1))
#     # print(target_class_pred)
#     # print(hard_pixels)
#     # print(hard_pixels_pred)
#     # print(hard_pixels_pred_log)
#     denom_term = tf.reduce_sum(tf.cast(hard_pixels_bool, dtype=y_pred.dtype))
#     numerator_term = -tf.reduce_sum(hard_pixels_pred_log)

#     loss = (numerator_term + smooth)/ (denom_term + smooth)
#     return loss

# def dice_basic(y_true, y_pred):
#     # y_true_f = K.flatten(y_true)
#     # y_pred_f = K.flatten(y_pred)
#     intersect = K.sum(y_true * y_pred)
#     denom = K.sum(y_true) + K.sum(y_pred)
#     return ((2. * intersect + 1e-10) / (denom + 1e-10))


# # 多分类时可计算某一类的dice值
# def dice_channel(y_true, y_pred, channel=0):
#     y_true = y_true[..., channel]
#     y_pred = y_pred[..., channel]
#     return dice_basic(y_true, y_pred)