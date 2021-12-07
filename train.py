import torch
import pathlib

from unet import UNet
from trainer import Trainer
from dataloader import dataloader_training, dataloader_validation
from plot_training import plot_training


# device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# model
model = UNet(in_channels=3,
             out_channels=2,
             n_blocks=4,
             start_filters=32,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=2).to(device)

# criterion
criterion = torch.nn.CrossEntropyLoss()
#criterion = torch.nn.BCELoss()
#criterion = torch.nn.BCEWithLogitsLoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# trainer
trainer = Trainer(model=model,
                  device=device,
                  criterion=criterion,
                  optimizer=optimizer,
                  training_DataLoader=dataloader_training,
                  validation_DataLoader=dataloader_validation,
                  lr_scheduler=None,
                  epochs=60,
                  epoch=0,
                  notebook=True)

# start training
training_losses, validation_losses, lr_rates = trainer.run_trainer()

model_name =  'flare_model.pt'
torch.save(model.state_dict(), pathlib.Path.cwd() / model_name)

fig = plot_training(training_losses,
                    validation_losses,
                    lr_rates,
                    gaussian=True,
                    sigma=1,
                    figsize=(10,4))