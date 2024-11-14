import pytorch_lightning as pl

class CheckPointCallback(pl.callbacks.ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        backbone_keys = [name for name, param in pl_module.named_parameters() if not param.requires_grad]
        for key in backbone_keys:
            del checkpoint['state_dict'][key]
        return checkpoint