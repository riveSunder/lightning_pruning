# Pruning with Lightning

A simple demo of incorporating pruning into a digits classifier using the `ModelPruning` callback from PyTorch Lightning.

```
git clone git@github.com:riveSunder/lightning_pruning.git
# or
#git clone https://github.com/riveSunder/lightning_pruning.git

cd lightning_pruning
  
virtualenv my_env --python=python3
source ~/my_env/bin/activate
pip install -r requirements.txt
```

## Run training

```
python demo/train.py
```

Or open the [notebook](notebooks/pruning_demo.ipynb) -> [in mybinder](https://mybinder.org/v2/gh/rivesunder/lightning_pruning/master?labpath=notebooks%2Fpruning_demo.ipynb) -> [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rivesunder/lightning_pruning/master?labpath=notebooks%2Fpruning_demo.ipynb), [in colab](https://colab.research.google.com/github/rivesunder/lightning_pruning/blob/master/notebooks/pruning_demo.ipynb) -> [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rivesunder/lightning_pruning/blob/master/notebooks/pruning_demo.ipynb) 
