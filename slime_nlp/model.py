import time, os
import matplotlib.pyplot as plt
from pandas import concat as pd_concat
import torch as pt
import torch.nn as nn
import torch.optim as optim
from torcheval.metrics.functional import binary_accuracy, binary_f1_score
from transformers import AutoConfig, AutoModel
from .dataset import CustomDset

plt.rcParams.update({"text.usetex": True, "font.family": "DejaVu Sans", "font.size": 14})


class CustomModel(nn.Module):
    """
    CustomModel: A transformer-based model for binary classification.

    Attributes:
        - pretrained_name (str): Pretrained transformer model name from the Hugging Face repository.
    """

    def __init__(self, pretrained_name="google-bert/bert-base-cased"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_name)
        config = AutoConfig.from_pretrained(pretrained_name)
        self.max_length = config.max_position_embeddings
        self.drop = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        Perform a forward pass on the input data.
        Args:
            input_ids (Tensor): Token IDs.
            token_type_ids (Tensor): Segment IDs for distinguishing sentence pairs.
            attention_mask (Tensor): Attention mask for padding tokens.
        Returns:
            Tensor: Linear classifier output.
        """
        x = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, 
                      attention_mask=attention_mask, return_dict=True)
        x = self.drop(x.pooler_output)
        return self.classifier(x)

    def load(self, path_name="weights/model_weights.pt", device='cpu'):
        """
        Load pre-trained weights into the model.
        Args:
            path_name (str): Path to the weights file.
            device (str): Device for loading (e.g., 'cpu' or 'cuda').
        """
        self.__device = device
        self.load_state_dict(pt.load(path_name, map_location=device))
        self.eval()

    def predict(self, data):
        """
        Predict binary labels for input data.
        Args:
            data (pd.DataFrame): DataFrame with 'text' (str) and 'group' (int) columns.
        Returns:
            Tensor: Predicted labels (0 or 1).
        """
        self.eval()
        dset = CustomDset(data, self.max_length, self.__device)
        pred = []
        for X, _ in dset:
            y_pred = self(*X).detach().cpu().sigmoid()
            pred.append(1 if y_pred >= 0.5 else 0)
        return pt.Tensor(pred)


class FitModel:
    """
    FitModel: Facilitates training and evaluation of CustomModel.

    Attributes:
        - device (str): Device for computations ('cpu' or 'cuda').
        - optimizer (str): Optimizer name (default: 'AdamW').
        - lr (float): Learning rate for the transformer weights.
        - lr_sub (float): Learning rate for classifier weights.
        - eps (float): Epsilon for optimizer stability.
    """

    def __init__(self, device='cpu', optimizer='AdamW', lr=2e-5, lr_sub=2e-4, eps=1e-8):
        self.lr = lr
        self.lr_sub = lr_sub
        self.eps = eps
        self._device = pt.device('cuda' if (device == 'cuda' and pt.cuda.is_available()) else 'cpu')
        self.optimizer = optimizer
        self.loss_fun = nn.BCEWithLogitsLoss()
        self.metric1 = binary_accuracy
        self.metric2 = binary_f1_score

    def train_step(self, X, y):
        """
        Perform a single training step.
        Args:
            X (Tensor): Input data.
            y (Tensor): Labels.
        Returns:
            Tensor: Loss value.
        """
        self.opt.zero_grad()
        loss = self.loss_fun(self.model(*X), y)
        loss.backward()
        self.opt.step()
        return loss

    def fit(self, train_data, val_data=None, epochs=1, batch_size=1, pretrained_name="google-bert/bert-base-cased", 
            klabel='', path_name=None, patience=0, min_delta=1e-2):
        """
        Train the model on the given data.
        """
        # Model and optimizer setup
        self.model = CustomModel(pretrained_name).to(self._device)
        max_length = AutoConfig.from_pretrained(pretrained_name).max_position_embeddings
        self.opt = getattr(optim, self.optimizer)([
            {"params": self.model.bert.parameters(), "lr": self.lr},
            {"params": self.model.classifier.parameters(), "lr": self.lr_sub}
        ], eps=self.eps)

        # Dataset setup
        train_dset = CustomDset(train_data, max_length, batch_size, self._device)
        val_dset = CustomDset(val_data, max_length, batch_size, self._device) if val_data is not None else None

        train_loss, val_metric1, val_metric2 = [], [], []

        for epoch in range(epochs):
            self.model.train()
            step_loss = []
            for X, y in train_dset:
                loss = self.train_step(X, y)
                step_loss.append(loss.item())
            train_loss.append(step_loss)

            if val_data is not None:
                self.model.eval()
                val = []
                with pt.no_grad():
                    for X, y in val_dset:
                        y_pred = self.model(*X).sigmoid().cpu()
                        val.append([y_pred, y])
                val_tensor = pt.tensor(val)
                val_metric1.append(self.metric1(*val_tensor.T))
                val_metric2.append(self.metric2(*val_tensor.T))

        self.train_loss = pt.tensor(train_loss)
        self.val_metric1 = pt.tensor(val_metric1)
        self.val_metric2 = pt.tensor(val_metric2)

    def save(self, path_name="weights/model_weights.pt"):
        """
        Save the model's state dictionary to the specified path.
        """
        os.makedirs(os.path.dirname(path_name), exist_ok=True)
        pt.save(self.model.state_dict(), path_name)
