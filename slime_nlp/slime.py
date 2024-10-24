import torch as pt
import pandas as pd

from transformers import AutoConfig, AutoTokenizer
from captum.attr import LayerIntegratedGradients

from matplotlib.colors import LinearSegmentedColormap, rgb2hex 
from IPython.display import display, HTML
from html2image import Html2Image

from .model import CustomModel

class ExplainModel:

    '''
    # ExplainModel: model explanability tools for data processing and visualization.
    
    Input: (model_name=None, device='cpu', n_steps=50, pretrained_name="google-bert/bert-base-cased")
    -----
    - model_name (str): string with the path and model's name.
    - device (str): select CPU or GPU device for output tensors.
    - n_steps (int): number of steps for Integrated Gradient approximation.
    - pretained_name (str): pretrained model name from huggingface.co repository.
    
    
    Methods:
    -------
    - explain: (text)
      -- text (str): text as string format.
    
      Returns a dictionary with 
      > input_ids (Tensor[int]): sequence of special tokens IDs.
      > token_list (List[str]): of tokens.
      > attributions (Tensor[float]): Integrated Gradient's attribution score by token.
      > delta (Tensor[float]): Integrated Gradient's error metric.
    
    - model_prediction: (input_ids)
      -- input_ids (Tensor): sequence of special tokens IDs.
    
      Returns a dictionary with
      > prob (float): classification probability score in [0, 1].
      > class (int): classification integer score 0 or 1.
    
    - visualize: (data, cmap_size=20, colors=["#73949e", "white", "#e2a8a7"], path_name=None)
      -- data (DataFrame): pandas dataframe with "text" and "group" columns.
      -- cmap_size (int): color-map discretization size.
      -- colors (List[str]): list of color in hex for color-map.
      -- path_name (str): string with the path and figure's name for output saving.
    
      Returns the tokenized text with attribution score by token.
    
    - attribution_by_token: (data, path_name=None, return_results=False):
      -- data (DataFrame): pandas dataframe with "id", "text", and "group" columns.
      -- path_name (str): string with the path and dataframe's name for saving.
      -- return_results (bool): boolean variable for returning dataframe.
    
      Returns a dataframe with 
      > id (str): text's ID.
      > condition (str): string to indicate "condition" or "control" group.
      > group (int): integer corresponding to the condition label (0 or 1).
      > pred_label (int): model's predition group (0 or 1).
      > score (float): the sum of the text's attribution values.
      > attribution (float): token's attribution value.
      > token (str): text's token.
    
    '''

    def __init__(self, model_name=None, device='cpu', n_steps=50, pretrained_name="google-bert/bert-base-cased"):

        if device == 'cpu':
            self._device = pt.device('cpu')
        
        elif device == 'gpu' or device == 'cuda': 
            self._device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')

        self.n_steps = n_steps

        # load model:
        self.model = CustomModel(pretrained_name).to(device)
        
        if model_name is not None:
            self.model.load(model_name, device)

        config = AutoConfig.from_pretrained(pretrained_name)
        self.max_length = config.max_position_embeddings
        
        # set Gradient Integrated:
        self.lig = LayerIntegratedGradients(self.model, self.model.bert.embeddings.word_embeddings)
        
        # get input_ids, baseline_ids, token_list:
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        
        self.baseline_token_id = self.tokenizer.pad_token_id 
        self.sep_token_id = self.tokenizer.sep_token_id 
        self.cls_token_id = self.tokenizer.cls_token_id 
        
    
    def explain(self, text):
        
        input_ids = self.tokenizer.encode(text, max_length=self.max_length, truncation=True, padding=True)        
        
        token_list = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        baseline_ids = [self.cls_token_id] + (len(input_ids)-2)*[self.baseline_token_id] + [self.sep_token_id]
        
        input_ids = pt.tensor(input_ids).unsqueeze(0).to(self._device)
        baseline_ids = pt.tensor(baseline_ids).unsqueeze(0).to(self._device)

        # get attributions:
        attributions, delta = self.lig.attribute(inputs=input_ids, baselines=baseline_ids, 
                                                 return_convergence_delta=True, n_steps=self.n_steps)

        attributions = attributions.detach().cpu()
        delta = delta.detach().cpu()
        
        # summarized attributions:
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions /= pt.norm(attributions)

        return {'input_ids':input_ids.cpu(), 'token_list':token_list, 
                'attributions':attributions, 'delta':delta}
        
    
    def model_prediction(self, input_ids):
        
        pred = self.model(input_ids)
        pred = pred[0,0].detach().cpu()
        pred_prob = pred.sigmoid()
        pred_class = 1 if pred_prob >= 0.5 else 0
    
        return {'prob':pred_prob.item(), 'class':pred_class}

    
    def visualize(self, data, cmap_size=20, colors=["#73949e", "white", "#e2a8a7"], path_name=None):

        texts = data['text'].tolist()
        groups = data['group'].tolist()
        
        # color range:
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=cmap_size)
        color_range = [rgb2hex(cmap(i)) for i in range(cmap_size)]

        for text, group in zip(texts, groups):
            exp = self.explain(text)
            pred = self.model_prediction(exp['input_ids'])
                        
            attr_by_token = ""
            for token, attr in zip(exp['token_list'], exp['attributions']):
                # attr = [-1:1]
                attr = (attr + 1)/2 # = [0:1]
                i = int(cmap_size*attr)
        
                attr_by_token += f" <span style='background-color: {color_range[i]}'>{token}</span>"
                
            
            html = ["<table width: 100%>"]
            html.append('<div style="border-top: 1px solid; margin-top: 5px; \
                         padding-top: 5px; display: inline-block">')
            
            html.append("<b>Legend: </b>")
    
            for color, label in zip(colors, ["Control", "Neutral", "Condition"]):                    
                html.append(f'<span style="display: inline-block; width: 10px; height: 10px;\
                             border: 1px solid; background-color: \
                             {color}" ></span> {label} ')

            html.append("</div>")
            columns = ["<tr><th>True Label</th>",
                       "<th>Predicted Label</th>",
                       "<th>Predicted probability</th>",
                       "<th>Attribution Score</th>"]
            
            html.append("".join(columns))

            results = [f"<tr><th>{'condition' if group == 1 else 'control'}</th>",
                       f"<th>{'condition' if pred['class'] == 1 else 'control'}</th>",
                       f"<th>{pred['prob']:.2f}</th>",
                       f"<th>{exp['attributions'].sum().item():.2f}</th>"]

            html.append("".join(results))
            html.append("</table>")
            html.append(attr_by_token)
            html.append("<br><br>")
            html = "".join(html)

            if path_name is not None:
                pn = path_name.split("/")
                path, name = "/".join(pn[:-1]), pn[-1]
                
                hti = Html2Image(size=(500, 400), output_path=path)            
                hti.screenshot(html_str=html, save_as=name)
                
            display(HTML(html))

    
    def attribution_by_token(self, data, path_name=None, return_results=False):
    
        N = len(data)
        output = pd.DataFrame()
        
        ids = data['id'].tolist()
        groups = data['group'].tolist()
        texts = data['text'].tolist()
        
        for n, (id, group, text) in enumerate(zip(ids, groups, texts)):
            
            exp = self.explain(text)
            pred = self.model_prediction(exp['input_ids'])
            
            score = exp['attributions'].sum().item()
            condition = "AD" if group == 1 else "control"
        
            result = {}
            for attr, token in zip(exp['attributions'], exp['token_list']):
                
                result['id'] = id
                result['condition'] = condition
                result['group'] = group
                result['pred_label'] = pred['class']
                result['score'] = score
                result['attributions'] = attr.item()
                result['token'] = token
                
                df_temp = pd.DataFrame([result])
                output = pd.concat([output, df_temp], ignore_index=True)
        
            print(f"Processing: {100*(n+1)/N:.1f}%", end='\r', flush=True)
        
        output = output.set_index("id")
        
        if path_name is not None:
            output.to_csv(path_name)
            
        if return_results: 
            return output

