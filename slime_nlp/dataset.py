import torch as pt
from transformers import AutoTokenizer
from pandas import read_csv as pd_csv

class ImportData:
    '''
    # ImportData: Imports a dataframe and splits it into train, validation, and test sets.

    Parameters:
    -----------
    - path_name (str): Path to the CSV file containing the dataset.
    - n_val (Optional[float]): Fraction of the data to reserve for validation (0 <= n_val < 1).
    - n_test (Optional[float]): Fraction of the data to reserve for testing (0 <= n_test < 1).
    - group_by (Optional[List[str]]): List of column names to subset the dataframe.
    - verbose (bool): If True, prints information about the dataset split.

    Attributes:
    -----------
    - df (DataFrame): Full dataset shuffled randomly.
    - train (DataFrame): Training subset of the dataframe.
    - val (Optional[DataFrame]): Validation subset of the dataframe, or None if `n_val` is not specified.
    - test (Optional[DataFrame]): Test subset of the dataframe, or None if `n_test` is not specified.

    Example Usage:
    --------------
    >>> data = ImportData("data.csv", n_val=0.1, n_test=0.1, group_by=["text", "label"])
    >>> print(len(data.train), len(data.val), len(data.test))
    '''
    def __init__(self, path_name, n_val=None, n_test=None, group_by=None, verbose=True):
        df = pd_csv(path_name)
        if group_by: 
            df = df[group_by]
        self.df = df.sample(frac=1)
        N = len(df)
        self.N_val = int(N * n_val) if n_val else 0
        self.N_test = int(N * n_test) if n_test else 0
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
        if self.N_val == 0: 
            return None
        return self.df[self.N_train:self.N_train + self.N_val]

    @property
    def test(self):
        if self.N_test == 0: 
            return None
        return self.df[self.N_train + self.N_val:self.N_train + self.N_val + self.N_test]


class CustomDset:
    '''
    # CustomDset: A PyTorch dataset for tokenized text sequences with labels.

    Parameters:
    -----------
    - data (DataFrame): Dataframe containing at least 'text' (str) and 'group' (int) columns.
    - max_length (int): Maximum length for tokenized sequences.
    - batch_size (int): Number of samples per batch.
    - shuffle (bool): Whether to shuffle the data before processing.
    - device (str): Device for the output tensors ('cpu', 'gpu', or 'cuda').
    - pretrained_name (str): HuggingFace model name for tokenizer initialization.

    Methods:
    --------
    - __len__: Returns the total number of samples in the dataset.
    - __getitem__: Fetches a batch of tokenized data and labels.

    Returns:
    --------
    - Tuple[Tuple[Tensor, Tensor, Tensor], Tensor]: 
        - (input_ids, token_type_ids, attention_mask): Tokenized tensors for input.
        - label: Corresponding labels as a tensor.

    Example Usage:
    --------------
    >>> from transformers import AutoTokenizer
    >>> dataset = CustomDset(data, max_length=128, batch_size=32)
    >>> for (inputs, label) in dataset:
    >>>     print(inputs[0].shape, label.shape)
    '''
    def __init__(self, data, max_length, batch_size=1, shuffle=True, device='gpu', pretrained_name="google-bert/bert-base-cased"):
        if shuffle:
            data = data.iloc[pt.randperm(len(data))]
        self.data = data
        self._device = pt.device('cuda' if (device in ['gpu', 'cuda'] and pt.cuda.is_available()) else 'cpu')
        self.max_length = max_length    
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[self.batch_size * index:self.batch_size * (index + 1)]['text'].tolist()
        label = self.data.iloc[self.batch_size * index:self.batch_size * (index + 1)]['group'].tolist()

        encoder = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, 
                                 truncation=True, padding=True)
        input_ids = encoder['input_ids'].to(self._device)
        token_type_ids = encoder['token_type_ids'].to(self._device)
        attention_mask = encoder['attention_mask'].to(self._device)
        label = pt.Tensor([label]).T.to(self._device)

        return (input_ids, token_type_ids, attention_mask), label
