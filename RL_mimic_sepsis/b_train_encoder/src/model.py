import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import pytorch_lightning as pl


class AIS_GRU(pl.LightningModule):
    """Model contains GRU for AIS. 
    """
    def __init__(self, state_dim, context_dim, action_dim, latent_dim, hidden_dim=128, lr=1e-3):
        """'hidden_dim': the dimension inside the model.
        'latent_dim': the dimension of the representation.

        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.gen = nn.Sequential(
            # Input: observation + demographic + previous action.
            nn.Linear(state_dim + context_dim + action_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.GRU(hidden_dim, latent_dim, batch_first=True),
            ExtractRNNOutput(),
        )
        self.pred = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        # TODO: What is weight_init?
        self.gen.apply(weight_init)
        self.pred.apply(weight_init)
    
    def forward(self, x):
        # We only need to get the state embedding, instead of predicted action.
        h = self.gen(x) 
        return h
    
    def _get_prediction_mse_loss(self, batch): # 
        """Calculate Mean Square Error of predicted observation and next obervation.
        An inner method starting from '_'. 
        Uses current (observation + demographic) and previous action to build representation.
        Then use representation + current action to predict next observation.
        """
        # Forward pass.
        curr_obs, curr_dem, prev_act, curr_act, next_obs, mask = self.process_batch(batch)
        curr_input = torch.cat([curr_obs, curr_dem, prev_act], dim=-1)
        curr_hidden = self.gen(curr_input)
        curr_ha = torch.cat([curr_hidden, curr_act], dim=-1)
        pred_obs = self.pred(curr_ha)

        mse = torch.mean(torch.square(next_obs - pred_obs)[~mask])
        return mse
    
    def _get_prediction_nll_loss(self, batch):
        curr_obs, curr_dem, prev_act, curr_act, next_obs, mask = self.process_batch(batch)
        
        # Forward pass.
        curr_input = torch.cat([curr_obs, curr_dem, prev_act], dim=-1)
        curr_hidden = self.gen(curr_input)
        curr_ha = torch.cat([curr_hidden, curr_act], dim=-1)
        pred_obs = self.pred(curr_ha)
        
        # Gaussian negative log likelihood loss.
        pred_dist = torch.distributions.MultivariateNormal(
            pred_obs, 
            torch.eye(pred_obs.shape[-1], device=self.device)
            )
        logprobs = pred_dist.log_prob(next_obs)
        logprobs[mask] = 0.0   # log probability of masked out timestep is log(1) = 0
        loss = -logprobs.sum(axis=1).mean()
        
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._get_prediction_nll_loss(batch)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        loss = self._get_prediction_nll_loss(batch)
        mse = self._get_prediction_mse_loss(batch)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_mse', mse, prog_bar=True, logger=True)
        return {'val_loss': loss, 'val_mse': mse}
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """For AIS MSE."""
        curr_obs, curr_dem, prev_act, curr_act, next_obs, mask = self.process_batch(batch)

        curr_input = torch.cat([curr_obs, curr_dem, prev_act], dim=-1)
        curr_hidden = self.gen(curr_input)
        pred_obs = self.pred(torch.cat([curr_hidden, curr_act], dim=-1))

        sq_err = (next_obs - pred_obs).pow(2)  

        # Expand mask to match feature dimension and filter out invalid elements
        valid_sq_err = sq_err[~mask.unsqueeze(-1).expand_as(sq_err)]  
        return valid_sq_err      
    
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_mse = torch.stack([x['val_mse'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss, 'avg_val_mse': avg_mse}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=30, 
            min_lr=1e-6, threshold=1e-3, verbose=True
            )
        return [
            [optimizer],
            [{
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }]
        ]
    
    def process_batch(self, batch):
        """Solves temporal misalignment of data. 
        'batch': a TensorDataset object that have 4 tensors.
        """
        dem, obs, act, lng = batch
        
        # Shift observations to make current-next observation pairs.
        # 'curr_obs': 1-step padding for each trajectory removed.
        # 'next_obs': the fist step of each trajectory removed.
        curr_obs, next_obs = obs[:, :-1, :], obs[:, 1:, :]
        curr_dem = dem[:, :-1, :]
        
        # Compute mask for extra appended rows of observations (all zeros along dim 2).
        mask = (next_obs == 0).all(dim=2)

        # Previous action is action observed at the current timestep.
        # Current action is the action of the next timestep.
        prev_act = act[:, :-1, :]
        curr_act = act[:, 1:, :]
        
        return curr_obs, curr_dem, prev_act, curr_act, next_obs, mask



class ExtractRNNOutput(nn.Module):
    def forward(self, x):
        out, _ = x
        return out


def weight_init(m):
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
