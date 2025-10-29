#!/usr/bin/env python3

#Import required packages
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

#Print available gpus, this will be saved to your output file
print(f"GPUs available: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

def main(batch_size, epochs, lr, num_workers, work_dir, output_dir):
    
    #Define datamodule
    class CustomDataModule(L.LightningDataModule):
        def __init__(self, batch_size=batch_size, num_workers=num_workers):
            super().__init__()
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
    
        def prepare_data(self):
            # Downloads only once
            datasets.FashionMNIST(root=work_dir+"/data", train=True, download=True)
            datasets.FashionMNIST(root=work_dir+"/data", train=False, download=True)
    
        def setup(self, stage=None):
            self.train_set = datasets.FashionMNIST(root=work_dir+"/data", train=True,
                                                   transform=self.transform)
            self.val_set = datasets.FashionMNIST(root=work_dir+"/data", train=False,
                                                 transform=self.transform)
    
        def train_dataloader(self):
            return DataLoader(self.train_set, batch_size=self.batch_size,
                              shuffle=True, num_workers=self.num_workers)
    
        def val_dataloader(self):
            return DataLoader(self.val_set, batch_size=self.batch_size,
                              shuffle=False, num_workers=self.num_workers)
    
    #Define pretrained resnet18 model
    class PretrainedResNet18(L.LightningModule):
        def __init__(self, lr=lr):
            super().__init__()
            self.save_hyperparameters()
            self.model = models.resnet18(weights="IMAGENET1K_V1")
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, 10)
            self.loss_fn = nn.CrossEntropyLoss()
    
        def forward(self, x):
            return self.model(x)
    
        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = self.loss_fn(y_hat, y)
            acc = (y_hat.argmax(dim=1) == y).float().mean()
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
            return loss
    
        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = self.loss_fn(y_hat, y)
            acc = (y_hat.argmax(dim=1) == y).float().mean()
            self.log("val_loss", loss, prog_bar=True)
            self.log("val_acc", acc, prog_bar=True)
    
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    #Instantiate data module and resnet model
    data_module = CustomDataModule()
    resnet_model = PretrainedResNet18()
    logger = CSVLogger(save_dir=output_dir, name="logs")
    
    #Create a checkpoint callback to save every 50 epochs
    #We are going to reference this in our trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="fashionmnist_resnet18_{epoch:03d}",
        save_top_k=-1,  
        every_n_epochs=50
    )
    
    #Define trainer
    #Since we won't be running this in a notebook, we change our strategy to "ddp"
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto", 
        precision="16-mixed" if torch.cuda.is_available() else "32",
        logger=logger,
        deterministic=False,
        log_every_n_steps=50,
        callbacks=[checkpoint_callback],
        default_root_dir=output_dir
    )
    
    #Train and validate our model
    trainer.fit(resnet_model, datamodule=data_module)
    trainer.validate(resnet_model, datamodule=data_module)
    
    #Save the final checkpoint
    trainer.save_checkpoint(os.path.join(output_dir, "fashionmnist_resnet18_final.ckpt"))

if __name__ == "__main__":

    #Define hyperparameters, here we switch to 150 epochs and 8 workers
    BATCH_SIZE = 64
    EPOCHS = 5
    LR = 1e-3
    NUM_WORKERS = 8
    WORK_DIR = "/insert/path/to/your/directory/postdoc_onboarding"
    OUTPUT_DIR = WORK_DIR + "/lightning_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #Run the pipeline
    main(batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR, 
         num_workers=NUM_WORKERS, work_dir=WORK_DIR, 
         output_dir=OUTPUT_DIR)
