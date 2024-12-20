import torch as pt
from transformers import AutoTokenizer
from pandas import read_csv as pd_csv

class ImportData:

    '''
    # ImportData: import dataframe and split it into train, validation, and test data.
    
    Input: (path_name, n_val=None, n_test=None, group_by=None, shuffle=True, verbose=True)
    -----
    - path_name (str): string with path and data name.
    - n_val (float): quantile of validation data.
    - n_test (float): quantile of test data.
    - group_by (List[str]): list of the dataframe's column names to group by.
    - shuffle (bool): boolean variable to allow dataframe shuffling.
    - verbose (bool): boolean variable to print dataset info.


    Attributes: 
    ----------
    - train (Dataframe): pandas dataframe of train batch.
    - val (Dataframe): pandas dataframe of validation batch.
    - test (Dataframe): pandas dataframe of test batch.
      
    '''
    
    def __init__(self, path_name, n_val=None, n_test=None, group_by=None, shuffle=True, verbose=True):

        df = pd_csv(path_name)

        if group_by: df = df[group_by]
            
        if shuffle: self.df = df.sample(frac=1)
        else: self.df = df
            
        N = len(df)
        
        if n_val: self.N_val = int(N*n_val)
        else: self.N_val = 0
            
        if n_test: self.N_test = int(N*n_test)
        else: self.N_test = 0
            
        self.N_train = N - self.N_val - self.N_test

        if verbose:
            print("DataFrame:\n", df.head(3))
            print(f"\nData length: N_total = {N}") 
            print(f"N-train = {self.N_train}, N-val = {self.N_val}, N-test = {self.N_test}\n")

    
    @property
    def train(self):
        return self.df[:self.N_train]
    
    @property
    def val(self):
        if self.N_val == 0: return None
        else: return self.df[self.N_train:self.N_train+self.N_val]

    @property
    def test(self):
        if self.N_test == 0: return None
        else: return self.df[self.N_train+self.N_val:self.N_train+self.N_val+self.N_test]
    

class CustomDset:

    '''
    # CustomDset: import the data sentences to return a PyTorch generator of tokenized 
    tensors.

    Input: (data, max_length, batch_size=1, shuffle=True, device='cpu',
            pretrained_name="google-bert/bert-base-cased")
    ----- 
    - data (Dataframe): pandas dataframe (ImportData's output) with "text"(str) and 
    "group"(int) columns.
    - max_length (int): the sequence maximum length.
    - batch_size (int): data batch-size value.
    - shuffle (bool): boolean variable for data shuffling.
    - device (str): select CPU or GPU device for output tensors.
    - pretained_name (str): pretrained model name from huggingface.co repository.


    Methods:
    -------
    __len__ (int): returns data size.
    
    __getitem__ (Tuple[Tensor, Tensor, Tensor], Tensor): generator 
    
    
    Output (generator): (input_ids, token_type_ids, attention_mask), label
    ------ 
    - input_ids (Tensor[int]): sequence of special tokens IDs.
    - token_type_ids (Tensor[int]): sequence of token indices to distinguish between 
    sentence pairs.
    - attention_mask (Tensor[int]): mask to avoid performing attention on padding token 
    indices.
    - label (Tensor): the corresponding label for the input sequence.
      
    '''
    
    def __init__(self, data, max_length, batch_size=1, shuffle=True, device='gpu', pretrained_name="google-bert/bert-base-cased"):

        if shuffle: data = data.iloc[pt.randperm(len(data))]
            
        self.data = data
        
        if device == 'cpu':
            self._device = pt.device('cpu')
        elif device == 'gpu' or device == 'cuda': 
            self._device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')

        self.max_length = max_length    
        self.batch_size = batch_size
        
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

    
    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, index):

        text = self.data.iloc[self.batch_size*index:self.batch_size*(index+1)]['text'].tolist()
        label = self.data.iloc[self.batch_size*index:self.batch_size*(index+1)]['group'].tolist()

        encoder = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, 
                                 truncation=True, padding=True)
        
        input_ids = encoder['input_ids'].to(self._device)
        token_type_ids = encoder['token_type_ids'].to(self._device)
        attention_mask = encoder['attention_mask'].to(self._device)
        label = pt.Tensor([label]).T.to(self._device)
        
        return (input_ids, token_type_ids, attention_mask), label

