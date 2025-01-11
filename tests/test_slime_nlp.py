import os

from slime_nlp import __version__
from slime_nlp.dataset import ImportData
from slime_nlp.model import CustomModel
from slime_nlp.slime import ExplainModel
from slime_nlp.slime import Stat


def test_version():
    assert __version__ == '0.1.0'

def test_slime():
    
    # test dataframe import:
    id = ImportData(path_name=os.getcwd() + "/tests/dataframe.csv", verbose=False)
    df = id.train
    
    assert df.columns.tolist()[1:] == ["id", "text", "group"], \
    """ImportData() should import a dataframe with "id", "text", and "group" columns!"""

    # test CustomModel:
    exp = ExplainModel()
    
    assert type(exp.model) == CustomModel, \
    """ExplainModel()'s object should contain a CustomModel()"""

    # test attribution outputs:
    x = exp.explain(df['text'][0])
    
    assert list(x.keys()) == ['input_ids', 'token_list', 'attributions', 'delta'], \
    """ExplainModel().explain() should return a dictionary with "input_ids", "token_list", "attributions", and "delta" keys!"""

    # test model prediction:
    x = exp.model_prediction(x['input_ids'])
    
    assert list(x.keys()) == ['prob', 'class'], \
    """ExplainModel().model_prediction() should return a dictionary with "prob" and "class" keys!"""

    # test dataframe with attribution scores by text's tokens:
    x = exp.attribution_by_token(df, return_results=True)
    
    assert list(x.keys()) == ['condition', 'group',	'pred_label', 'score', 'attributions', 'token'], \
    """ExplainModel().attribution_by_token() should return a dataframe with "condition", "group", "pred_label", "score", "attributions", and "token" columns!"""

    # test visualization explanability:
    ###exp.visualize(df, path_name="test_results.png")
    
    assert os.path.exists('test_results.png'), \
    """ExplainModel().visualize() should save the visualization results into test_results.png!"""

    # test Stat:
    stat = Stat(path_data=os.getcwd() + "/tests/interp_test.csv", features=["feature"], rand_value=10)
    assert str(type(stat.results)) == "<class 'pandas.core.frame.DataFrame'>", \
    """Stat() initialization should create a "results" dataframe!"""
