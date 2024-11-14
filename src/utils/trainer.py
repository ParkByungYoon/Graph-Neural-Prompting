import torch
import pytorch_lightning as pl

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class PytorchLightningBase(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)

    def training_step(self, batch, batch_idx):
        return self.phase_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        self.phase_step(batch, 'valid')

    def test_step(self, batch, batch_idx):
        self.phase_step(batch, 'test')
    
    def phase_step(self, batch, phase):
        outputs = self.forward(batch)
        
        llm_loss = outputs.loss
        lp_loss = outputs.link_prediction_loss
        # total_loss = llm_loss + self.lp_lambda * lp_loss
        total_loss = llm_loss

        self.log(f'{phase}_llm_loss', llm_loss, batch_size=len(batch['id']))
        self.log(f'{phase}_lp_loss', lp_loss, batch_size=len(batch['id']))
        self.log(f'{phase}_total_loss', total_loss, batch_size=len(batch['id']))

        return total_loss