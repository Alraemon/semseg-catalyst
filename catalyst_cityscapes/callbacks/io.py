from pathlib import Path

import imageio
import numpy as np

from catalyst.dl import Callback, CallbackOrder, State, utils

from .utils import mask_to_overlay_image
import os
import matplotlib.pyplot as plt
from skimage import measure
from matplotlib.patches import Rectangle

from .loss_func import iou_safe, intersectionAndUnionGPU


class MeanIoU_Epoch(Callback):
    def __init__(
        self,
        prefix: str, 
        input_key: str = "mask",
        output_key: str = "pred_final",
        classes: int = 19,
        ignore_index: int = 255,
    ):
        super().__init__(CallbackOrder.Internal)
        self.prefix = prefix
        self.input_key = input_key
        self.output_key = output_key
        self.ignore_index = ignore_index
        self.classes = classes
        self.area_intersection = None
        self.area_union = None
        self.nb_samples = 0

    def _reset_stats(self):
        """Resets the confusion matrix holding the epoch-wise stats."""
        self.area_intersection = None
        self.area_union = None
        self.nb_samples = 0

    def on_batch_end(self, state: State):
        targets = state.batch_in[self.input_key]
        outputs = state.batch_out[self.output_key]
        
        self.nb_samples += targets.shape[0]

        area_intersection, area_union, area_target = intersectionAndUnionGPU(outputs, targets, self.classes, 
                                                    ignore_index=self.ignore_index)
        if self.area_intersection is None:
            self.area_intersection = area_intersection
            self.area_union = area_union
        else:
            self.area_intersection += area_intersection
            self.area_union += area_union
    
    def on_loader_end(self, state: State, eps = 1e-9):
        
        mIoU = (self.area_intersection + (self.area_union ==0) * eps ) / (self.area_union  + eps)
        
        self._reset_stats()
        


class OriginalImageSaverCallback(Callback):
    def __init__(
        self,
        output_dir: str,
        relative: bool = True,
        filename_suffix: str = "",
        filename_extension: str = ".jpg",
        input_key: str = "image",
        outpath_key: str = "filename",
    ):
        super().__init__(CallbackOrder.Logging)
        self.output_dir = Path(output_dir)
        self.relative = relative
        self.filename_suffix = filename_suffix
        self.filename_extension = filename_extension
        self.input_key = input_key
        self.outpath_key = outpath_key

    def get_image_path(self, state: State, name: str, suffix: str = ""):
        out_dir = (
            state.logdir / self.output_dir if self.relative else self.output_dir
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        # print('check1:', out_dir)
        name = flat_path_id(name)
        res = out_dir /str(f"{name}{suffix}{self.filename_extension}")
        # print('check2:', str(res))
        return res

    def on_batch_end(self, state: State):
        names = state.batch_in[self.outpath_key]
        images = state.batch_in[self.input_key]

        images = utils.tensor_to_ndimage(images.detach().cpu(), dtype=np.uint8)
        for image, name in zip(images, names):
            fname = self.get_image_path(state, name, self.filename_suffix)
            imageio.imwrite(fname, image)


def flat_path_id(path_w_subdir):
    source, pid, studyid, seriesid, slice_index =  path_w_subdir.split(os.sep)[-5:] # name = subdir/###.png
    # print(fp_chunks)
    short_fn = '_'.join([source, pid, studyid[-6:], seriesid[-6:], slice_index.replace('.png', '')])
    return short_fn


class OverlayMaskImageSaverCallback(OriginalImageSaverCallback):
    def __init__(
        self,
        output_dir: str,
        relative: bool = True,
        mask_strength: float = 0.5,
        filename_suffix: str = "",
        filename_extension: str = ".jpg",
        input_key: str = "image",
        input_target_key: str = 'mask',
        output_key: str = "pred",
        outpath_key: str = "filename",
    ):
        super().__init__(
            output_dir=output_dir,
            relative=relative,
            filename_suffix=filename_suffix,
            filename_extension=filename_extension,
            input_key=input_key,
            outpath_key=outpath_key,
        )
        self.mask_strength = mask_strength
        self.output_key = output_key
        self.input_target_key = input_target_key
        self.dice_list = []


    def _reset_stats(self):
        """Resets the confusion matrix holding the epoch-wise stats."""
        self.dice_list = []

    def on_batch_end(self, state: State):
        names = state.batch_in[self.outpath_key]
        images = state.batch_in[self.input_key]
        gtmasks = state.batch_in[self.input_target_key]

        images = utils.tensor_to_ndimage(images.detach().cpu())
        gtmasks = gtmasks.detach().cpu().squeeze(1).numpy()

        predmasks = state.batch_out[self.output_key]

        # print('img %s, gt %s, pred %s' %(images.shape, gtmasks.shape, predmasks.shape))
        for name, image, gtmask, predmask in zip(names, images, gtmasks, predmasks):
            # image = mask_to_overlay_image(image, mask, self.mask_strength)
            dice = dice_score(gtmask, predmask)
            self.dice_list.append(dice)
            plot_obj = PlotSeg2Image(image[..., 1])
            plot_obj.put_on_mask(gtmask, color= 'r', is_bbox_on=False)
            plot_obj.put_on_mask(predmask, color= 'g', is_bbox_on= False)
            fname = self.get_image_path(state, name, '%0.4f' %dice)
            # print('check2:', str(fname))
            plot_obj.save_fig(fname)
        
    
    def on_loader_end(self, state: State):
        """@TODO: Docs. Contribution is welcome.

        Args:
            state (State): current state
        """
        mDice_sample = sum(self.dice_list) / len(self.dice_list)
        print('Slices: %d, mean Dice %0.4f' %(len(self.dice_list), mDice_sample))
        self._reset_stats()


            # imageio.imwrite(fname, image)

def dice_score(gt, mask, epsilon = 1e-5):

    nominator = np.sum(gt * mask) * 2 + epsilon
    denominator = np.sum(gt + mask) + epsilon
    return nominator /denominator

class PlotSeg2Image(object):

    def __init__(self, image):
        fig = plt.figure(figsize=(16, 8))
        # plot ct image on the left as reference
        ax_ref = fig.add_subplot(121)
        ax_ref.imshow(image, cmap='gray')
        ax_ref.set_xticks([]), ax_ref.set_yticks([])
        self.fig = fig
        self.ax_ref = ax_ref

        # plot ct image on the right with other contours
        self.ax_mask = fig.add_subplot(122)
        self.ax_mask.imshow(image, cmap='gray')

    def put_on_mask(self, mask, color  = 'r', is_bbox_on = True):
        #skimage findcontours生成的，rows, cols
        contours = measure.find_contours(mask, 0.5, fully_connected='low', positive_orientation='low')
        self.put_on_edge(contours, color, is_bbox_on= is_bbox_on)

    def put_on_edge(self, countours, color = 'r', is_col_first = False, is_bbox_on = True):
        xy_ixs = [0, 1] if is_col_first else [1, 0]
        for contour in countours:
            contour = np.array(contour, dtype=np.int16)
            self.ax_mask.plot(contour[:, xy_ixs[0]], contour[:, xy_ixs[1]], color, linewidth=0.8)

            if is_bbox_on:
                contour = contour[..., ::(1 if is_col_first else -1)]
                r_min, c_min = np.min(contour, axis = 0)
                r_max, c_max = np.max(contour, axis = 0)
                self.put_on_bbox([c_min, r_min, c_max, r_max])

        self.ax_mask.set_xticks([]), self.ax_mask.set_yticks([])
        # return self.ax_mask

    def put_on_bbox(self, bbox, color = 'g'):
        # Create a Rectangle patch
        x1, y1, x2, y2 = bbox
        # lower left
        bottom_left_coord, width, height = (x1, y1), x2-x1,y2-y1
        # print(bottom_left_coord, width, height)
        rect = Rectangle(bottom_left_coord, width, height,linewidth=0.5, edgecolor=color,facecolor='none')
        # Add the patch to the Axes
        self.ax_mask.add_patch(rect)
        return self.ax_mask


    def save_fig(self, pic_save_path):
        self.pic_save_path = pic_save_path
        if pic_save_path is not None:
            plt.savefig(pic_save_path, bbox_inches='tight'), plt.close(self.fig)




__all__ = ["OriginalImageSaverCallback", "OverlayMaskImageSaverCallback"]
