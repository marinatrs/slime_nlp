from setuptools import setup

with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()

setup(name='SLIME-NLP',
      version='0.1.0',
      author=['Marina Ribeiro', 'Tibério Pereira'],
      description='Statistical and Linguistic Insights for Model Explanation',
      packages=['slime_nlp'],
      classifiers=["Programming Language :: Python :: 3",
                   "Operating System :: OS Independent"], 
      #install_requires=install_requires
)

