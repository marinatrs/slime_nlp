# __Statistical and Linguistic Insights for Model Explanation - SLIME__ 

__SLIME__ (_Statistical and Linguistic Insights for Model Explanation_) is an explainable method designed to identify lexical components in speech that are representative of neurological disorders, such as Alzheimer's Disease (AD), and to clarify their importance in a Large Language Model (LLM) decisions. This methodology employs a pipeline combining Integrated Gradients (IG) and statistical analysis to reveal how an LLM classifies speech as indicative of AD or control groups. By highlighting features such as reductions in social references common in AD, SLIME enhances the interpretability and accuracy of LLMs, providing a reliable tool for studying neurodegeneration and increasing confidence in applying LLMs in clinical neurological contexts.

<p float="central">
  <img src="figs/exp_S118.png" width="400"/>
  <img src="figs/exp_S177.png" width="400"/>  
</p>

## Installation

To install SLIME, we can use pip **command**:

```bash
pip install slime_nlp
```

## Content

The project is composed of three main [codes](https://github.com/marinatrs/slime_nlp/tree/main/slime_nlp):
- __dataset.py__ for pre-processing _.csv_ dataset;
- __model.py:__ the custom LLM for classification;
- __slime.py:__ for model explanability tools.

Check the tutorials in [docs](https://github.com/marinatrs/slime_nlp/tree/main/docs).


## Reference

The paper is currently in the submission process.

### About the authors
- [[ORCID] Marina Ribeiro](https://orcid.org/0000-0002-2516-3135)
- [[ORCID] Tibério Pereira](https://orcid.org/0000-0003-1856-6881)