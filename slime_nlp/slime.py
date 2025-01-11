import torch as pt
import pandas as pd
import numpy as np

from transformers import AutoConfig, AutoTokenizer
from captum.attr import LayerIntegratedGradients

from sklearn import metrics
from statsmodels.distributions.empirical_distribution import ECDF

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
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
      -- path_name (str): string with path and dataframe's names for saving.
      -- return_results (bool): boolean variable for returning dataframe.
    
      Returns a dataframe with 
      > id (str): text's ID.
      > condition (str): string to indicate "condition" or "control" group.
      > group (int): integer corresponding to the condition label (0 or 1).
      > pred_label (int): model's prediction group (0 or 1).
      > score (float): the sum of the text's attribution values.
      > attribution (float): token's attribution value.
      > token (str): token.

    - stat: (path_data, features, rand_value=5000)
      -- path_data (str): string with path and dataset name. This file is the user-dependent tagger output, 
      containing columns for tokens and associated features.
      -- features (List): list of features processed by the user-dependent tagger for visualization. Use Ellipsis (...)
      for considering an specific feature and its following ones, e.g, features=["BigWords", ...].
      -- rand_value (int): number of random subsamples of self.data. 
      -- path_results (str): string with path and dataframe results' name for saving in .csv file.
      -- return_results (bool): Boolean variable for returning the dataframe results.
    
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

        '''
        - explain: (text)
          -- text (str): text as string format.
        
          Returns a dictionary with 
          > input_ids (Tensor[int]): sequence of special tokens IDs.
          > token_list (List[str]): of tokens.
          > attributions (Tensor[float]): Integrated Gradient's attribution score by token.
          > delta (Tensor[float]): Integrated Gradient's error metric.
        
        '''
        
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

        '''
        - model_prediction: (input_ids)
          -- input_ids (Tensor): sequence of special tokens IDs.
        
          Returns a dictionary with
          > prob (float): classification probability score in [0, 1].
          > class (int): classification integer score 0 or 1.
        
        '''
        
        pred = self.model(input_ids)
        pred = pred[0,0].detach().cpu()
        pred_prob = pred.sigmoid()
        pred_class = 1 if pred_prob >= 0.5 else 0
    
        return {'prob':pred_prob.item(), 'class':pred_class}

    
    def visualize(self, data, cmap_size=20, colors=["#73949e", "white", "#e2a8a7"], path_name=None):

        '''
        - visualize: (data, cmap_size=20, colors=["#73949e", "white", "#e2a8a7"], path_name=None)
          -- data (DataFrame): pandas dataframe with "text" and "group" columns.
          -- cmap_size (int): color-map discretization size.
          -- colors (List[str]): list of color in hex for color-map.
          -- path_name (str): string with the path and figure's name for output saving.
        
          Returns the tokenized text with attribution score by token.
        
        '''
        
        ids = data['id'].tolist()
        texts = data['text'].tolist()
        groups = data['group'].tolist()
        
        # color range:
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=cmap_size)
        color_range = [rgb2hex(cmap(i)) for i in range(cmap_size)]

        for ID, text, group in zip(ids, texts, groups):
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

                n = name.split(".")
                name = "".join(n[:-1]) + f"_{ID}." + n[-1]
                                
                hti = Html2Image(size=(500, 400), output_path=path)            
                hti.screenshot(html_str=html, save_as=name)
                
            display(HTML(html))

    
    def attribution_by_token(self, data, path_name=None, return_results=False):

        '''
        - attribution_by_token: (data, path_name=None, return_results=False):
          -- data (DataFrame): pandas dataframe with "id", "text", and "group" columns.
          -- path_name (str): string with path and dataframe's names for saving.
          -- return_results (bool): boolean variable for returning dataframe.
        
          Returns a dataframe with 
          > id (str): text's ID.
          > condition (str): string to indicate "condition" or "control" group.
          > group (int): integer corresponding to the condition label (0 or 1).
          > pred_label (int): model's prediction group (0 or 1).
          > score (float): the sum of the text's attribution values.
          > attribution (float): token's attribution value.
          > token (str): token.
    
        '''
        
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


class Stat:

    '''
    # Stat: Statistical analysis for token attributions.

    Input: (path_data, features, rand_value=5000, path_results=None)
    -----
    - path_data (str): string with path and dataset name. This file is the user-dependent tagger output, 
    containing columns for tokens and associated features.
    - features (List): list of features processed by the user-dependent tagger statistical computation. 
    Use Ellipsis (...) for considering an specific feature and its following ones, e.g, 
    features=["BigWords", ...].
    - rand_value (int): number of random subsamples of data. 
    - path_results (str): string with path and dataframe results' name for saving in .csv file.


    Methods:
    -------
    - plot_dist: (features, path_plot=None)
      -- features (List): list of features processed by the user-dependent tagger for visualization. Use 
      Ellipsis (...) for considering an specific feature and its following ones, e.g, 
      features=["BigWords", ...].
      -- path_plot (str): string with path and plots' name for saving in .png file.

      Returns distribution density plots of feature attributions across all tokens to assess group 
      relevance.

    - plot_scatter: (path_plot=None)
      -- path_plot (str): string with path and plots' name for saving in .png file.

      Returns a plots with all linguistic features distinguishing from the AUC median of random subsamples.

    - plot_bars: (path_plot=None)
      -- path_plot (str): string with path and plots' name for saving in .png file.

      Returns a bar plot as an alternative visualization from the scatter plot.
            
    '''

    def __init__(self, path_data, features, rand_value=5000, path_results=None):
        
        data = pd.read_csv(path_data)
    
        if features[-1] is Ellipsis: user_features = data.loc[:, features[0]:]
        else: user_features = data.loc[:, features]
        
        data = pd.concat([data.id, data.group, data.attribution, user_features], axis=1)
        
        by_feature = {} 
        Dict = {'feature':0, 'mean_attr':0, 'realp':0, 'attribution':0, 'attribute':0, 
                'fpr':0, 'tpr':0, 'mean_attr_rand':0, 'auc_random_dist':0}

        results = {'feature':[], 'AUC_impact':[], 'AUC':[], 'AUC_random':[], 'AUC_diff':[], 
                   'group':[], 'attribution':[], 'count':[], 'percentile':[]}
        
        for i in user_features:

            by_feature[i] = Dict
            results['feature'].append(i)
            
            attribute = data.attribution.where(data[i] > 0)

            by_feature[i]['attribute'] = attribute
            by_feature[i]['attribution'] = data.attribution
            
            results['count'].append(np.sum(data[i]))
            
            mask = data[i]            
            mean_attr_rand = [np.mean(data.attribution.where(np.random.permutation(mask) > 0)) 
                              for _ in range(rand_value)]
            
            mean_attr = np.mean(data.attribution.where(mask > 0))

            by_feature[i]['mean_attr_rand'] = mean_attr_rand
            by_feature[i]['mean_attr'] = mean_attr            
            results['attribution'].append(mean_attr)
            
            if (mean_attr > np.percentile(mean_attr_rand, 95)) or (mean_attr < np.percentile(mean_attr_rand, 5)):
                if mean_attr <= 0: results['group'].append('control')
                else: results['group'].append('condition')
            else:
                results['group'].append('none')

            data['temp'] = data.attribution*mask
            
            mean = data.groupby(['id']).mean()['temp'].to_numpy()
            median = data.groupby(['id']).median()['group'].to_numpy()
            
            fpr, tpr, thresholds = metrics.roc_curve(median, mean)
            by_feature[i]['fpr'] = fpr
            by_feature[i]['tpr'] = tpr
            
            realp = metrics.auc(fpr, tpr)
            by_feature[i]['realp'] = realp
            results['AUC'].append(realp)
                        
            auc_random_dist = np.zeros(rand_value)
            
            for j in range(rand_value):
                data['temp'] = data.attribution*np.random.permutation(mask)
                mean = data.groupby(['id']).mean()['temp'].to_numpy()
                
                fpr, tpr, thresholds = metrics.roc_curve(median, mean)
                auc_random_dist[j] = metrics.auc(fpr, tpr)

            by_feature[i]['auc_random_dist'] = auc_random_dist
            
            results['percentile'].append(ECDF(auc_random_dist)(realp))
            results['AUC_random'].append(np.percentile(auc_random_dist, 50))
            results['AUC_diff'].append(results['AUC'][-1] - results['AUC_random'][-1])
            
            if (realp > np.percentile(auc_random_dist, 95)):
                results['AUC_impact'].append('positive')
                
            elif (realp < np.percentile(auc_random_dist, 5)):
                results['AUC_impact'].append('negative')
            else: 
                results['AUC_impact'].append('none')
        
        results = pd.DataFrame(results)
        
        if path_results is not None:
            results.to_csv(path_results, index = False)  
            
        self.by_feature = by_feature
        self.results = results

    
    def plot_dist(self, features, path_plot=None):

        '''
        - plot_dist: (features, path_plot=None)
          -- features (List): list of features processed by the user-dependent tagger for visualization. Use 
          Ellipsis (...) for considering an specific feature and its following ones, e.g, 
          features=["BigWords", ...].
          -- path_plot (str): string with path and plots' name for saving in .png file.
    
          Returns distribution density plots of feature attributions across all tokens to assess group 
          relevance.
    
        '''
        
        for i in features:
            
            fig, axs = plt.subplots(2, 2, figsize=(14, 12))
            
            sns.set_context(context='paper', font_scale=1.6)
        
            # 1st plot: Density plot of feature attribution and random distribution
            sns.kdeplot(self.by_feature[i]['attribute'], ax=axs[0, 0], color='#d33932', label=i)
            sns.kdeplot(self.by_feature[i]['attribution'], ax=axs[0, 0], color='black', linestyle='dashed', 
                        label='All')
            axs[0, 0].set_xlabel('Sample Attribution')
            axs[0, 0].set_ylabel('Density')
            axs[0, 0].set_frame_on(False)
            axs[0, 0].legend(loc='upper right')   
        
            # 2nd plot: Histogram of average attribution of random distributions
            axs[0, 1].hist(self.by_feature[i]['mean_attr_rand'], bins=30, color='#424243', edgecolor='white', 
                           label='Random Permutations')
            axs[0, 1].axvline(self.by_feature[i]['mean_attr'], color='#d33932', linewidth=3, label=i)
            axs[0, 1].set_xlabel('Average Attribution')
            axs[0, 1].set_ylabel('Count')
            axs[0, 1].set_frame_on(False)
            axs[0, 1].legend(loc='best')  
            
            # 3rd plot: ROC curve of condition vs control
            axs[1, 0].plot(self.by_feature[i]['fpr'], self.by_feature[i]['tpr'], color='#d33932', lw=3, 
                           label=f'Control vs Condition (AUC = {self.by_feature[i]['realp']:.2f})')
            axs[1, 0].plot([0, 1], [0, 1], 'k--', label='Chance level (AUC = 0.5)')  # Diagonal line
            axs[1, 0].set_xlabel('False Positive Rate')
            axs[1, 0].set_ylabel('True Positive Rate')
            axs[1, 0].fill_between(self.by_feature[i]['fpr'], self.by_feature[i]['tpr'], facecolor='#d33932', 
                                   alpha=.3)
            axs[1, 0].set_frame_on(False)
            axs[1, 0].legend(loc='lower right')
             
            # 4th plot: Histogram of AUC values of random distributions
            axs[1, 1].hist(self.by_feature[i]['auc_random_dist'], bins=100, color='#424243', edgecolor='white', 
                           label='Random permutations')
            axs[1, 1].axvline(self.by_feature[i]['realp'], color='#d33932', lw=3, label=i)
            axs[1, 1].axvline(np.percentile(self.by_feature[i]['auc_random_dist'], 50), color='#efe74e', lw=3, 
                              label='(median)')
            axs[1, 1].set_xlabel('AUC')
            axs[1, 1].set_ylabel('Count')
            axs[1, 1].set_frame_on(False)
            axs[1, 1].legend(loc='best')    
            
            # Final adjustments
            plt.tight_layout()
            plt.rcParams['pdf.fonttype'] = 42 
        
            if path_plot is not None: 
                n = path_plot.split(".")
                path_plot = ".".join(n[:-1]) + f"_{i}." + n[-1]
                plt.savefig(path_plot)     
            else: 
                plt.show()


    def plot_scatter(self, path_plot=None):

        '''
        - plot_scatter: (path_plot=None)
          -- path_plot (str): string with path and plots' name for saving in .png file.
    
          Returns a plots with all linguistic features distinguishing from the AUC median of random subsamples.

        '''
        
        plt.figure(figsize=(14, 9), facecolor='white')
        sns.set_theme(context='talk', style='white')
        
        markers = {"control": "o", "none": "s", "condition": "X"}
        custom_palette = {'positive': 'green', 'negative': 'gray', 'none': 'gray'}
        
        sns.scatterplot(data=self.results, x='attribution', y='AUC_diff', hue='AUC_impact', style="group",
                        markers=markers, palette=custom_palette, s=200)
        
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.axvline(0, color='gray', linestyle='--', linewidth=1)
        
        plt.annotate('Control', xy=(self.results['attribution'].min(), 0), xycoords='data', 
                     xytext=(-70, -5), textcoords='offset points', ha='center', color='gray')
        
        plt.annotate('Condition', xy=(self.results['attribution'].max(), 0), xycoords='data', 
                     xytext=(80, -5), textcoords='offset points', ha='center', color='gray')
        
        plt.annotate('Help classification', xy=(0, self.results['AUC_diff'].max()), xycoords='data', 
                     xytext=(5, 40), textcoords='offset points', ha='center', color='gray')
        
        plt.annotate('Worsen classification', xy=(0, self.results['AUC_diff'].min()), xycoords='data', 
                     xytext=(5, -40), textcoords='offset points', ha='center', color='gray')
        
        labels = self.results.feature
        for i in range(len(self.results.feature)):
            if self.results.AUC_impact[i] == 'positive':
                plt.annotate(labels[i], (self.results.attribution[i], self.results.AUC_diff[i]), 
                             textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.xlabel('Attribution')
        plt.ylabel('AUC Difference')
        
        plt.grid(False)
        plt.axis('off')
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Significant increase of AUC', markerfacecolor='green', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Control', markerfacecolor='gray', markersize=10),
            Line2D([0], [0], marker='X', color='w', label='Condition', markerfacecolor='gray', markersize=10),
            Line2D([0], [0], marker='s', color='w', label='None', markerfacecolor='gray', markersize=10)
        ]
        
        plt.legend(handles=legend_elements, loc='best', frameon=False)
        plt.rcParams['pdf.fonttype'] = 42
        
        if path_plot is not None: plt.savefig(path_plot, dpi=300)
        else: plt.show()

    
    def plot_bars(self, path_plot=None):

        '''
        - plot_bars: (path_plot=None)
          -- path_plot (str): string with path and plots' name for saving in .png file.
    
          Returns a bar plot as an alternative visualization from the scatter plot.

        '''
        
        sns.set_theme(context='talk', style='white') 
        
        def get_color(row):
            base_color = {'condition': '#e2a8a7', 'none': '#7b7b7b', 'control': '#73949e'}
            alpha = 1.0 if row['AUC_impact'] == 'positive' else 0.7
            return base_color[row['group']], alpha
    
        patches = [mpatches.Patch(color='#e2a8a7', label='condition'),
                   mpatches.Patch(color='#7b7b7b', label='none'),
                   mpatches.Patch(color='#73949e', label='control'),
                   mpatches.Patch(facecolor = 'white', edgecolor='black', label='Significant AUC increase')]
        
        fig, ax = plt.subplots(1, 2, figsize=(10,12)) 
        
        filtered_data = self.results[(self.results['attribution'] != 0) & (~self.results['attribution'].isna())]
        N = len(filtered_data)
        
        for i in [0, 1]:
        
            sorted_data = filtered_data.sort_values(by='attribution', ascending=False)
            sorted_data = sorted_data[i*N//2:(i+1)*N//2]
            
            colors, alphas = zip(*sorted_data.apply(get_color, axis=1))
            
            bars = sns.barplot(data=sorted_data, y='feature', x='attribution', ax=ax[i])
            
            for bar, color, alpha in zip(bars.patches, colors, alphas):
                bar.set_color(color)
                bar.set_alpha(alpha)
    
                edge_color = 'black' if alpha == 1.0 else 'gray'
                
                bar.set_edgecolor(edge_color)
                bar.set_linewidth(1)  
    
            if i == 1:
                ax[i].yaxis.tick_right()
                ax[i].tick_params(right=False)
                
            bars.set(xlabel='', ylabel='') 
            bars.set(xticks=[]) 
    
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['bottom'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['left'].set_visible(False)
        
        plt.legend(handles=patches, bbox_to_anchor=(0.0, 0.6))
        
        plt.rcParams['pdf.fonttype'] = 42
        plt.tight_layout()
                
        if path_plot is not None: plt.savefig(path_plot, dpi=300)
        else: plt.show()
