
## Create a virtual environment (optional)

This step is highly recommended. This way you wont have to deal with dependency and versioning issues.

```
python -m venv .venv
source ./.venv/bin/activate
```

## Install requirements

```
pip install -r requirements.txt
```

## Create nnunet dataset from Prostate158

```
python create_nnunet_dataset.py
```

Now you will have `nnUNet_raw` folder. By default it creates a dataset for t2 axial images as input and anatomy masks as output.

Now you can see [t2-to-anatomy-20-epoch.ipynb](t2-to-anatomy-20-epoch.ipynb) file to see how to train a fold.