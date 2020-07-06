import torch

from catalyst.dl import Callback, CallbackOrder, State

from .utils import encode_mask_with_color


class RawMaskPostprocessingCallback(Callback):
    def __init__(
        self,
        threshold: float = 0.5,
        input_key: str = "logits",
        output_key: str = "mask",
    ):
        super().__init__(CallbackOrder.Internal)
        self.threshold = threshold
        self.input_key = input_key
        self.output_key = output_key

    def on_batch_end(self, state: State):
        output = state.batch_out[self.input_key]

        if output.shape[1] == 1:
            output = (torch.sigmoid(output) > self.threshold).squeeze(1)
        else:
            output = output.max(1)[1]
        
        output = output.int().detach().cpu().numpy() #torch.sigmoid
        state.batch_out[self.output_key] = output
        # encode_mask_with_color(
            # output, self.threshold
        # )


__all__ = ["RawMaskPostprocessingCallback"]
