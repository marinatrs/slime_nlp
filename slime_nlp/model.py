import time, os
import matplotlib.pyplot as plt
from pandas import concat as pd_concat

import torch as pt
import torch.nn as nn
import torch.optim as optim
from torcheval.metrics.functional import binary_accuracy, binary_f1_score
from transformers import AutoConfig, AutoModel

from .dataset import CustomDset

plt.rcParams.update({"text.usetex":True, "font.family":"DejaVu Sans", "font.size":14})

class CustomModel(nn.Module):

    '''
    # CustomModel: Custom LLM for classification

    Input: (pretrained_name="google-bert/bert-base-cased")
    ----- 
    - pretained_name (str): pretrained model name from huggingface.co repository.

    Returns object with callable model's input.

    
    Methods:
    -------
    - forward = __call__: (input_ids, token_type_ids=None, attention_mask=None)
      -- input_ids (Tensor[int]): sequence of special tokens IDs.
      -- token_type_ids (Tensor[int]): sequence of token indices to distinguish 
      between sentence pairs.
      -- attention_mask (Tensor[int]): mask to avoid performing attention on padding 
      token indices.

      Returns a Tensor with linear prediction output.
    
    - load: (path_name="weights/model_weights.pt", device='cpu') 
      Loads model's weights.
      
      -- path_name (str): string with path and name of the model's weights (.pt)
      for saving.
      -- device (str): select CPU or GPU for prediction processing.
    
    - predict: (data)
      -- data (Dataframe): pandas dataframe (ImportData's output) with "text"(str) 
      and "group"(int) columns.

      Returns (Tensor[float]) a tensor with prediction scalars (0, 1).
    
    '''
    
    def __init__(self, pretrained_name="google-bert/bert-base-cased"):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(pretrained_name)
        config = AutoConfig.from_pretrained(pretrained_name)

        self.max_length = config.max_position_embeddings
        
        self.drop = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, 1)

    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        
        x = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, 
                      attention_mask=attention_mask, return_dict=True)

        x = x.pooler_output 

        x = self.drop(x)
        
        x = self.classifier(x)
        
        return x

    
    def load(self, path_name="weights/model_weights.pt", device='cpu'):

        self.__device = device 
        self.load_state_dict(pt.load(path_name, map_location=device))
        self.eval()
        
        
    def predict(self, data):

        '''
        - predict: (data)
          -- data (Dataframe): pandas dataframe (ImportData's output) with "text"(str) 
          and "group"(int) columns.

          Returns (Tensor[float]) a tensor with prediction scalars (0, 1).
        '''
        
        self.eval()
            
        dset = CustomDset(data, max_length=self.max_length, device=self.__device)
    
        pred = []
        for X, y in dset:        
            y_pred = self(*X)
            y_pred = y_pred.detach().cpu().sigmoid()
            y_pred = 1 if y_pred >= 0.5 else 0
            pred.append(y_pred)
        
        return pt.Tensor(pred)


class FitModel:

    '''
    # FitModel: CustomModel model fitting.

    Input: (device='cpu', optimizer='AdamW', lr=2e-5, lr_sub=2e-4, eps=1e-8)
    -----
    - device (str): select CPU or GPU for training.
    - optimizer (str): training optimizer name.
    - lr (float): learning-rate for AutoModel's LLM weights adjustment.
    - lr_sub (float): learning-rate for weights adjustment of the CustomModel's 
    additional layer block.
    - eps (float): optimizer constant for numerical stability.

    Methods:
    -------
    - train_step (X, y):
      -- X (Tensor): CustomModel input data.
      -- y (Tensor): tensor of numerical labels.

      Returns (Tensor) the loss function value.

    - fit (train_data, val_data=None, epochs=1, batch_size=1, pretrained_name="google-bert/bert-base-cased",
    klabel='', path_name=None, patience=0, min_delta=1e-2):
      -- train_data (Dataframe): pandas dataframe (ImportData's output) with "text"(str) 
      and "group"(int) columns.
      -- val_data (Dataframe): equivalent to train_data.
      -- epochs (int): number of epochs for training.
      -- batch_size (int): data batch-size value.
      -- pretained_name (str): pretrained model name from huggingface.co repository.
      -- klabel (str): string argument for kfold() method.
      -- path_name (srt): path and model name string for saving.
      -- patience (int): number of epochs to wait for the early-stop mechanism. For
      patience=0, the early-stop is not considered.
      -- min_delta (float): tolerance value above the best metric result during the training. 
      If no improvement is achieved, the training is stopped.

    - kfold (data, K=2, epochs=1, batch_size=1, model_name=None, pretrained_name="google-bert/bert-base-cased"):
      -- data (Dataframe): pandas dataframe (ImportData's output) with "text"(str) 
      and "group"(int) columns.
      -- K (int): number of folds for cross-validation.
      -- epochs (int): number of epochs for each cross-validation loop.
      -- batch_size (int): data batch-size value.
      -- model_name (srt): path and model name string for saving.
      -- pretained_name (str): pretrained model name from huggingface.co repository.

    - plot_metric (path_name=None):
      -- path_name (srt): path and plot name string for saving.
      
      Returns the plot of the loss function values and metrics during training.
      
    - fold_evaluation:
      Evaluates the performance of the K-fold cross-validation.

    - evaluate (data):
      -- data (Dataframe): pandas dataframe (ImportData's output) with "text"(str) 
      and "group"(int) columns.

      Returns the mean values of the F1-score and Accuracy metrics. 

    - save (path_name="weights/model_weights.pt"):
      -- model_name (srt): path and model name string for saving.

      Creates the "weights" directory and saves the model's inner parameters.
    
    '''
    
    def __init__(self, device='gpu', optimizer='AdamW', lr=2e-5, lr_sub=2e-4, eps=1e-8):

        self.lr = lr
        self.lr_sub = lr_sub
        self.eps = eps

        if device == 'cpu':
            self._device = pt.device('cpu')
        
        elif device == 'gpu' or device == 'cuda': 
            self._device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
        
        self.optimizer = optimizer                
        
        self.loss_fun = nn.BCEWithLogitsLoss()

        self.metric1 = binary_accuracy 
        self.metric2 = binary_f1_score
        
    
    def train_step(self, X, y):
        
        self.opt.zero_grad()
        
        # forward pass - train step
        y_pred = self.model(*X)

        loss = self.loss_fun(y_pred, y)
        
        # backward pass
        loss.backward()

        # Update parameters
        self.opt.step()
        
        return loss
        

    def fit(self, train_data, val_data=None, epochs=1, batch_size=1, pretrained_name="google-bert/bert-base-cased",
            klabel='', path_name=None, patience=0, min_delta=1e-2):

        '''
        - fit (train_data, val_data=None, epochs=1, batch_size=1, pretrained_name="google-bert/bert-base-cased",
        klabel='', path_name=None, patience=0, min_delta=1e-2):
          -- train_data (Dataframe): pandas dataframe (ImportData's output) with "text"(str) 
          and "group"(int) columns.
          -- val_data (Dataframe): equivalent to train_data.
          -- epochs (int): number of epochs for training.
          -- batch_size (int): data batch-size value.
          -- pretained_name (str): pretrained model name from huggingface.co repository.
          -- klabel (str): string argument for kfold() method.
          -- path_name (srt): path and model name string for saving.
          -- patience (int): number of epochs to wait for the early-stop mechanism. For
          patience=0, the early-stop is not considered.
          -- min_delta (float): tolerance value above the best metric result during the training. 
          If no improvement is achieved, the training is stopped.
          
        '''
        # set model:
        self.model = CustomModel(pretrained_name).to(self._device)
        config = AutoConfig.from_pretrained(pretrained_name)
        max_length = config.max_position_embeddings
        
        # set optimizer:
        opt_params = [{"params":self.model.bert.parameters(), "lr":self.lr}, 
                      {"params":self.model.classifier.parameters(), "lr":self.lr_sub}]

        self.opt = getattr(optim, self.optimizer)(opt_params, eps=self.eps)

        # set data iterators:
        self.dset_info = {"max_length":max_length, 
                          "batch_size":batch_size,
                          "device":self._device.type,
                          "pretrained_name":pretrained_name}
        
        train_dset = CustomDset(train_data, **self.dset_info)
        if val_data is not None: val_dset = CustomDset(val_data, **self.dset_info)
        else: val_dset = [0]
        
        self.epochs = epochs
        N_train = len(train_dset)
        N_val = len(val_dset)
        
        best, wait = 0.0, 0        
        train_loss, val_metric1, val_metric2 = [], [], []

        for epoch in range(epochs):

            # training block:
            self.model.train()

            print(f"#{klabel}Epoch {epoch+1}/{epochs}:")
            start_time = time.time()

            step_loss = []
            for step, (X, y) in enumerate(train_dset):

                loss = self.train_step(X, y)
                
                step_loss.append(loss.item())

                if step % 10 == 0:
                    print(f"Batch:{int(100*step/N_train)}%", end=' - ')
                    print(f"train-loss = {loss.item():.3e}", end='\r')

            train_loss.append(step_loss)
            
            print(f"Batch:{int(100*step/N_train)}%", end=' - ')
            print(f"<train-loss> = {pt.mean(pt.Tensor(train_loss[-1])):.3e}", end='\r', flush=True)
            print()            
            
            # validation block:
            if val_data is not None:
                self.model.eval()

                val = []
                with pt.no_grad():
                    for X, y in val_dset:                 
                        y_pred = self.model(*X)
                        y_pred = y_pred[0,0].detach().cpu().sigmoid()
                        y_pred = 1.0 if y_pred >= 0.5 else 0.0 
                        
                        val.append([y_pred, y[0,0]])
        
                val = pt.tensor(val)
                
                val_metric1.append(self.metric1(*val.T))
                val_metric2.append(self.metric2(*val.T))
                
                print("<validation-metric>:", end=" ")
                print(f"Acc = {val_metric1[-1]:.3e}", end=", ")
                print(f"F1 = {val_metric2[-1]:.3e}")
                print(f"Time taken: {time.time() - start_time:.2f}s\n")

                # early stop:
                if patience > 0: wait += 1
                    
                last_acc = val_metric1[-1]
                
                if last_acc > best:
                    best = last_acc
                    wait = 0

                elif last_acc > (best + min_delta):
                    wait = 0

                if wait > patience:
                    break

        self.train_loss = pt.Tensor(train_loss).mean(axis=1)
        self.val_metric1 = pt.Tensor(val_metric1)
        self.val_metric2 = pt.Tensor(val_metric2)

        if path_name is not None:
            self.save(path_name)
            
            
    def kfold(self, data, K=2, epochs=1, batch_size=1, model_name=None, pretrained_name="google-bert/bert-base-cased"):

        '''
        - kfold (data, K=2, epochs=1, batch_size=1, model_name=None, pretrained_name="google-bert/bert-base-cased"):
          -- data (Dataframe): pandas dataframe (ImportData's output) with "text"(str) 
          and "group"(int) columns.
          -- K (int): number of folds for cross-validation.
          -- epochs (int): number of epochs for each cross-validation loop.
          -- batch_size (int): data batch-size value.
          -- model_name (srt): path and model name string for saving.
          -- pretained_name (str): pretrained model name from huggingface.co repository.
          
        '''
        
        N = len(data)
        N_val = N//K
        
        print(f"Data size: Train={N-N_val} - Validation={N_val}\n")

        path_name = None
        k_train_loss, k_val_metric1, k_val_metric2 = [], [], []
        
        for k in range(K):
            klabel = f"{k+1}-Fold, "
            
            if model_name is not None: 
                path_name = f"weights/{model_name}_weights_{k+1}.pt"
                   
            val_data = data.iloc[k*N_val:(k+1)*N_val]
            
            train_data = pd_concat([data.iloc[:k*N_val], data.iloc[(k+1)*N_val:]])
            
            self.fit(train_data, val_data, epochs, batch_size, pretrained_name, klabel, path_name)
        
            k_train_loss.append(self.train_loss.tolist())
            k_val_metric1.append(self.val_metric1.tolist())
            k_val_metric2.append(self.val_metric2.tolist())

        self.train_loss = pt.Tensor(k_train_loss)
        self.val_metric1 = pt.Tensor(k_val_metric1)
        self.val_metric2 = pt.Tensor(k_val_metric2)
        
        self.fold_evaluation()

    
    def plot_metric(self, path_name=None):

        '''
        - plot_metric (path_name=None):
          -- path_name (srt): path and plot name string for saving.
          
          Returns the plot of the loss function values and metrics during training.
            
        '''
        
        val_size = self.val_metric1.shape[0]
        epochs = pt.arange(1, self.train_loss.shape[-1] + 1)
    
        if self.train_loss.dim() == 1:    
            if val_size:
                fig, ax = plt.subplots(1, 3, figsize=(18,3))
                
                ax[0].plot(epochs, self.train_loss)
                ax[1].plot(epochs, self.val_metric1)
                ax[2].plot(epochs, self.val_metric2)
                
                ax[0].set_ylabel("Loss function")
                ax[1].set_ylabel("Accuracy")
                ax[2].set_ylabel("F1 score")
        
                for i in range(3):
                    ax[i].set_xlabel("Epochs")
                    ax[i].grid()
        
            else:
                plt.figure(figsize=(6,4))
                plt.plot(epochs, self.train_loss)
                plt.ylabel("Loss function")
                plt.xlabel("Epochs")
                plt.grid()
    
        else: # K-fold cross-validation
            val_mean1 = self.val_metric1.mean(1)
            val_mean2 = self.val_metric2.mean(1)
            
            acc_std, acc_mean = pt.std_mean(val_mean1)
            f1_std, f1_mean = pt.std_mean(val_mean2)
            
            print(f"\nValidation scores over K-folds:", end=' ')
            print(f"Acc = ({acc_mean:.2f} +- {acc_std:.3f})", end=', ')
            print(f"F1 = ({f1_mean:.2f} +- {f1_std:.3f})\n")
    
            fig, ax = plt.subplots(1, 3, figsize=(18,3))
            zip_loss = zip(self.train_loss, self.val_metric1, self.val_metric2)
            
            for k, (train_loss, val_metric1, val_metric2) in enumerate(zip_loss, start=1):
                print(f"{k}-fold: <train-loss> = {pt.mean(train_loss):.2f}", end=", ")
                print(f"<val-acc> = {pt.mean(val_metric1):.2f}", end=", ")
                print(f"<val-F1> = {pt.mean(val_metric2):.2f}")
    
                ax[0].plot(epochs, train_loss, label=f"{k}-fold: train")
                ax[1].plot(epochs, val_metric1, label=f"{k}-fold: val")
                ax[2].plot(epochs, val_metric2, label=f"{k}-fold: val")
        
            ax[0].set_ylabel("Loss function")
            ax[1].set_ylabel("Accuracy")
            ax[2].set_ylabel("F1 score")
    
            for i in range(3):
                ax[i].set_xlabel("Epochs")
                ax[i].legend(fontsize=12)
                ax[i].grid()
        
        if path_name is not None: 
            plt.savefig(path_name, bbox_inches='tight')
            
            n = path_name.split(".")                
            pt.save(self.train_loss, n[0] + "_train-loss.pt")
            if val_size:
                pt.save(self.val_metric1, n[0] + "_val-metric1.pt")
                pt.save(self.val_metric2, n[0] + "_val-metric2.pt")                
            
        else: plt.show()

    
    def fold_evaluation(self):

        '''
        - fold_evaluation:
          Evaluates the performance of the K-fold cross-validation.
    
        '''
    
        val_mean1 = self.val_metric1.mean(1)
        val_mean2 = self.val_metric2.mean(1)
        
        acc_std, acc_mean = pt.std_mean(val_mean1)
        f1_std, f1_mean = pt.std_mean(val_mean2)
        
        print(f"\nValidation scores over K-folds:", end=' ')
        print(f"Acc = ({acc_mean:.2f} +- {acc_std:.3f})", end=', ')
        print(f"F1 = ({f1_mean:.2f} +- {f1_std:.3f})\n")

        best_fold = 1 + pt.argmax(val_mean1).item()
        path_dir = "./weights/"
        
        for f in os.listdir(path_dir):
            try: k = int(f.split("_")[-1].split(".")[0])
            except: continue
                
            if k == best_fold:
                s = f.split("_")
                f_best = "best_" + "_".join(s[:-1]) + "." + s[-1].split(".")[-1]
                os.rename(path_dir+f, path_dir+f_best)
            
            else: 
                os.remove(path_dir+f)
                    
    
    def evaluate(self, data):

        '''
        - evaluate (data):
          -- data (Dataframe): pandas dataframe (ImportData's output) with "text"(str) 
          and "group"(int) columns.
    
          Returns the mean values of the F1-score and Accuracy metrics. 

        '''
        
        dset = CustomDset(data, **self.dset_info)
        
        val = []
        for X, y in dset:        
            y_pred = self.model(*X)
            y_pred = y_pred[0, 0].detach().cpu().sigmoid()
            y_pred = 1.0 if y_pred >= 0.5 else 0.0
            
            val.append([y_pred, y[0, 0]])
    
        val = pt.tensor(val)
    
        acc = self.metric1(*val.T)
        f1 = self.metric2(*val.T)
        
        print(f"Accuracy = {acc:.3f}")
        print(f"F1-score = {f1:.3f}")
    
    
    def save(self, path_name="weights/model_weights.pt"):

        '''
        - save (path_name="weights/model_weights.pt"):
          -- model_name (srt): path and model name string for saving.
    
          Creates the "weights" directory and saves the model's inner parameters.
    
        '''
        
        if not os.path.exists("./weights/"): 
        	os.makedirs("./weights/") 

        pt.save(self.model.state_dict(), path_name)

