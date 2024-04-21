# ATCS Practical 1
## Structure 
`data/` is the folder in which the vocab and embeddings will be stored.
`models/` contains the pytorch files for the trained models.
`NLI/` contains the source code for this project.
`runs/` contains the tensorboards for the various models.
`SentEval/` contains the senteval files and the data for it.
`environment.yml` contains the conda env for this project.
`NLI/results.ipynb` shows the final results as well as some analysis.

## Prepare project
1. Make use of conda and paste the following command: `conda env create -f environment.yml` 
2. Download the pretrained models from the url and paste them into the `models/` folder. https://drive.google.com/drive/folders/1qJDkg35UgpBe9OW_T6ZHTFVTwI3j5SMN?usp=sharing
3. Also the glove embeddings can be found in the url, make sure to place this file in the `data/` folder.
4. For evaluating on SentEval: 
    1. Make sure to download the SentEval github repo and place the folder in this project, extract and rename to SentEval
    2. Run the following command to install SentEval: `pip install git+https://github.com/facebookresearch/SentEval.git`
    3. Move into the `data/downstream` folderthen run the following command `bash SentEval/data/downstream/get_transfer_data.bash`   
    4. **IMPORTANT**: there is a bug in the SentEval code. When SentEval is installed through the conda env, open the `utils.py` file. This file can be found in this location `C:\Users\<USERNAME>\anaconda3\envs\ATCS_6\Lib\site-packages\senteval`. And then comment out lines 89 through 93.

## Training a model
If you want to train a model from scratch, use the following command: `python NLI\train.py --model <MODELNAME>` The options for the models are: baseline, LSTM, BILSTM, BILSTM_MAX.

## Evaluating the model
If you want to evaluate the model, use the following command: `python NLI\eval.py --MODEL <MODELNAME>` with the same options as above.

