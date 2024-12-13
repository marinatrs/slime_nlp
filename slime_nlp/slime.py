import torch as pt
import pandas as pd
import numpy as np

from transformers import AutoConfig, AutoTokenizer
from captum.attr import LayerIntegratedGradients

from sklearn import metrics
from statsmodels.distributions.empirical_distribution import ECDF

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, rgb2hex 
from IPython.display import display, HTML
from html2image import Html2Image

from .model import CustomModel

class ExplainModel:

    """
    ExplainModel: A toolkit for model explainability, data processing, and visualization.

    Parameters:
    ----------
    model_name : str, optional
        Path and name of the model to load. Default is None.
    device : str, optional
        Device to run the model and computations on ('cpu', 'cuda', or 'gpu'). Defaults to 'cpu'.
    n_steps : int, optional
        Number of steps for the Integrated Gradients approximation. Default is 50.
    pretrained_name : str, optional
        Pretrained model identifier from Hugging Face's model hub. Defaults to "google-bert/bert-base-cased".

    Methods:
    -------
    explain(text: str) -> Dict[str, Any]
        Generates token-level attributions for the input text using Integrated Gradients.

    model_prediction(input_ids: Tensor) -> Dict[str, Any]
        Produces the model's prediction probabilities and class labels.

    visualize(data: pd.DataFrame, cmap_size: int = 20, colors: List[str] = [...], path_name: str = None)
        Visualizes token-level attributions for a dataset and saves the results as images.

    attribution_by_token(data: pd.DataFrame, path_name: str = None, return_results: bool = False) -> pd.DataFrame
        Computes and returns a DataFrame with token-level attributions and optional saved output.

    stat(data_path: str, features: List[str], rand_value: int = 5000, results_path: str = None, return_results: bool = False) -> pd.DataFrame
        Performs statistical analysis on feature-level attributions and saves the results if specified.
    """

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


    @staticmethod
    def stat(data_path, features, rand_value=5000, results_path=None, return_results=False):
        
        data = pd.read_csv(data_path)
        
        if features[-1] is Ellipsis: user_features = data.loc[:, features[0]:]
        else: user_features = data.loc[:, features]
        
        data = pd.concat([data.id, data.group, data.attribution, user_features], axis=1)
    
        feature, AUC_impact, AUC, group, attr, count, percentile, AUC_random = [], [], [], [], [], [], [], []
        
        for i in user_features:
                   
            feature.append(i)
            
            attribute = data.attribution.where(data[i] > 0)
            
            count.append(np.sum(data[i]))
        
            mask = data[i]
            
            mean_attr_rand = [np.mean(data.attribution.where(np.random.permutation(mask) > 0)) 
                           for _ in range(rand_value)]
            
            mean_attr = np.mean(data.attribution.where(mask > 0))
            
            attr.append(mean_attr)
            
            if (mean_attr > np.percentile(mean_attr_rand, 95)) or (mean_attr < np.percentile(mean_attr_rand, 5)):
                if mean_attr <= 0: group.append('control')
                else: group.append('condition')    
            else:
                group.append('none')
        
            data['temp'] = data.attribution*mask
            
            mean = data.groupby(['id']).mean()['temp'].to_numpy()
            median = data.groupby(['id']).median()['group'].to_numpy()
            
            fpr, tpr, thresholds = metrics.roc_curve(median, mean)
            
            realp = metrics.auc(fpr, tpr)
    
            AUC.append(realp)
            
            auc_random_dist = np.zeros(rand_value)
            
            for j in range(rand_value):
                data['temp'] = data.attribution*np.random.permutation(mask)
                
                mean = data.groupby(['id']).mean()['temp'].to_numpy()
                
                fpr, tpr, thresholds = metrics.roc_curve(median, mean)
                
                auc_random_dist[j] = metrics.auc(fpr, tpr)
            
            AUC_random.append(np.percentile(auc_random_dist, 50))
            percentile.append(ECDF(auc_random_dist)(realp))
            
            if (realp > np.percentile(auc_random_dist, 95)):
                AUC_impact.append('positive')
            elif (realp < np.percentile(auc_random_dist, 5)):
                AUC_impact.append('negative')
            else: 
                AUC_impact.append('none')
      
            fig, axs = plt.subplots(2, 2, figsize=(14, 12))
            
            sns.set_context(context='paper', font_scale=1.6)
    
            # 1st plot: Density plot of feature attribution and random distribution
            sns.kdeplot(attribute, ax=axs[0, 0], color='#d33932', label=i)
            sns.kdeplot(data.attribution, ax=axs[0, 0], color='black', linestyle='dashed', label='All')
            axs[0, 0].set_xlabel('Sample Attribution')
            axs[0, 0].set_ylabel('Density')
            axs[0, 0].set_frame_on(False)
            axs[0, 0].legend(loc='upper right')   
    
            # 2nd plot: Histogram of average attribution of random distributions
            axs[0, 1].hist(mean_attr_rand, bins=30, color='#424243', edgecolor='white', label='Random Permutations')
            axs[0, 1].axvline(mean_attr, color='#d33932', linewidth=3, label=i)
            axs[0, 1].set_xlabel('Average Attribution')
            axs[0, 1].set_ylabel('Count')
            axs[0, 1].set_frame_on(False)
            axs[0, 1].legend(loc='best')  
            
            # 3rd plot: ROC curve of condition vs control
            axs[1, 0].plot(fpr, tpr, color='#d33932', lw=3, label=f'Control vs Condition (AUC = {realp:.2f})')
            axs[1, 0].plot([0, 1], [0, 1], 'k--', label='Chance level (AUC = 0.5)')  # Diagonal line
            axs[1, 0].set_xlabel('False Positive Rate')
            axs[1, 0].set_ylabel('True Positive Rate')
            axs[1, 0].fill_between(fpr, tpr, facecolor='#d33932', alpha=.3)
            axs[1, 0].set_frame_on(False)
            axs[1, 0].legend(loc='lower right')
             
            # 4th plot: Histogram of AUC values of random distributions
            axs[1, 1].hist(auc_random_dist, bins=100, color='#424243', edgecolor='white', label='Random permutations')
            axs[1, 1].axvline(realp, color='#d33932', lw=3, label=i)
            axs[1, 1].axvline(np.percentile(auc_random_dist, 50), color='#efe74e', lw=3, label='(median)')
            axs[1, 1].set_xlabel('AUC')
            axs[1, 1].set_ylabel('Count')
            axs[1, 1].set_frame_on(False)
            axs[1, 1].legend(loc='best')    
            
            # Final adjustments
            plt.tight_layout()
            plt.rcParams['pdf.fonttype'] = 42 
            plt.show()
            
        df_results = pd.DataFrame({'feature': feature, 'AUC_impact': AUC_impact, 'AUC': AUC, 
                                   'AUC_random': AUC_random, 'percentile': percentile, 
                                   'count': count, 'group': group, 'attribution': attr})
    
        if results_path is not None:
            df_results.to_csv(results_path, index = False)
            
        if return_results: return df_results
            