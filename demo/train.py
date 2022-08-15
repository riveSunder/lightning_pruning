import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelPruning

import sklearn
import sklearn.datasets

def get_digits():

    dataset = sklearn.datasets.load_digits()

    data_x = dataset["data"]
    data_y = dataset["target"]

    default_dtype = torch.get_default_dtype()

    data_x = torch.tensor(data_x, dtype=default_dtype)
    data_y = torch.tensor(data_y, dtype=torch.long)

    return data_x, data_y

def get_compute_pruning_rate(schedule=[(100, 0.5)]):
    
    def compute_pruning_rate(epoch):
        for threshold_epoch, rate in schedule:
            if epoch == threshold_epoch:
                msg = f"prune at rate {rate} epoch {epoch}"
                print(msg)
                return rate

    return compute_pruning_rate

class DigitsModel(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()

        self.learning_rate = kwargs["lr"] \
                if "lr" in kwargs.keys() else 3e-4
        self.dim_in = kwargs["dim_in"] \
                if "dim_in" in kwargs.keys() else 64
        self.number_classes = kwargs["number_classes"]\
                if "number_classes" in kwargs.keys() else 10
        self.dim_h = kwargs["dim_h"]\
                if "dim_h" in kwargs.keys() else 256
        self.l2_penalty = kwargs["l2"] \
                if "l2" in kwargs.keys() else 0.0
        self.dropout_rate = kwargs["dropout_rate"]\
                if "dropout_rate" in kwargs.keys() else 0.0

        

        self.layer_0 = nn.Linear(self.dim_in, self.dim_h)
        self.layer_1 = nn.Linear(self.dim_h, self.dim_h)
        self.layer_2 = nn.Linear(self.dim_h, self.dim_h)
        self.layer_3 = nn.Linear(self.dim_h, self.dim_h)
        self.layer_4 = nn.Linear(self.dim_h, self.dim_h)
        self.layer_out = nn.Linear(self.dim_h, self.number_classes)

    def forward(self, x):
        
        x = torch.relu(self.layer_0(x))
        x = torch.relu(self.layer_1(x))
        x = F.dropout(x, p=self.dropout_rate, \
                training=self.training)
        x = torch.relu(self.layer_2(x))
        x = F.dropout(x, p=self.dropout_rate, \
                training=self.training)
        x = torch.relu(self.layer_3(x))
        x = F.dropout(x, p=self.dropout_rate, \
                training=self.training)
        x = torch.relu(self.layer_4(x))
        x = F.dropout(x, p=self.dropout_rate, \
                training=self.training)

        output = self.layer_out(x)

        return output

    def training_step(self, batch, batch_idx):
        
        data_x, targets = batch[0], batch[1]

        predictions = self.forward(data_x)

        loss = F.cross_entropy(predictions, targets)
        accuracy = torchmetrics.functional.accuracy(predictions, targets)
        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        
        data_x, targets = batch[0], batch[1]

        predictions = self.forward(data_x)

        validation_loss = F.cross_entropy(predictions, targets)
        validation_accuracy = torchmetrics.functional.accuracy(predictions, targets)
        self.log("val_loss", validation_loss)
        self.log("val_accuracy", validation_accuracy)
    

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), \
                lr=self.learning_rate, \
                weight_decay=self.l2_penalty)

        return optimizer
    
    def count_params(self):

        count = 0

        for p in self.parameters():
            count += p.numel() 

        return count

    def count_pruned(self):

        count_active = 0

        for name, p in self.named_parameters():
            
            count_active += torch.sum(p != 0.0)

        total_count = self.count_params()
        prune_msg = f"{count_active} nonzero of"\
                f" {total_count} parameters, "\
                f"{count_active/total_count:.4e}"
        print(prune_msg)
            

if __name__ == "__main__":

    max_epochs = 256
    num_workers = 2
    batch_size = 128
    dropout_rate = 0.5
    l2 = 1e-6
    lr=1e-4
    dim_h = 1024
    my_seeds = [1, 13, 42] 


    data_x, target = get_digits()


    for use_pruning in [True, False]:


        if use_pruning:
            pruning_schedule = [(elem, 0.495) \
                    for elem in range(100,201,33)]
        else:
            pruning_schedule = []


        for my_seed in my_seeds:    
            np.random.seed(my_seed)
            torch.manual_seed(my_seed)


            model = DigitsModel(dropout_rate=dropout_rate, \
                    dim_h=dim_h, l2=l2, lr=lr)

            model(data_x)

            test_x, test_y = data_x[-100:], target[-100:]
            dataset = TensorDataset(data_x[:400], target[:400]) 
            val_dataset = TensorDataset(\
                    data_x[-400:-100], target[-400:-100]) 
            train_dataloader = DataLoader(dataset, \
                    batch_size=batch_size, \
                    num_workers=num_workers)
            val_dataloader = DataLoader(val_dataset, \
                    batch_size=batch_size, \
                    num_workers=num_workers)


            if torch.cuda.is_available():
                trainer = pl.Trainer(accelerator="gpu", \
                        devices=1, max_epochs=max_epochs,\
                        callbacks=[ModelPruning("l1_unstructured",\
                        amount=get_compute_pruning_rate())])
            else:
                trainer = pl.Trainer(max_epochs=max_epochs,\
                        callbacks=[ModelPruning("l1_unstructured",\
                        amount=get_compute_pruning_rate(\
                        schedule=pruning_schedule))])

            model.count_pruned()

            dummy_input = data_x[:32] * 0.0

            onnx_name = f"pre_s{my_seed}_p{use_pruning}.onnx"
            torch.onnx.export(model, dummy_input, \
                    onnx_name, verbose=True)
            trainer.fit(model=model, \
                    train_dataloaders=train_dataloader,\
                    val_dataloaders=val_dataloader)

            onnx_name = f"post_s{my_seed}_p{use_pruning}.onnx"

            torch.onnx.export(model, dummy_input, \
                    onnx_name, verbose=True)

            model.count_pruned()




    import pdb; pdb.set_trace()
