# PhosAF
#PhosAF: an integrated deep learning archi-tecture for predicting protein phosphoryla-tion sites based on feature fusion with AlphaFold2
##System requirement
PhosAF is developed under Windows environment with:
numpy  1.21.6
pandas  0.23.4
tensorflow-gpu  2.2.0

##model
The input file is an csv file, which includes proteinname, sequence and label. Among them, proteinname contains the information of position and residue and the format of proteinname is '>sp|name|residue|pisotion'. For example, '>sp|P06748|S|48'. Besides, label is 0 or 1, represents non-phoshphorylation and phoshphorylation site.
###Train your own data
If you want to train your own data, please prepare your train data as a csv file, which contain three columns: proteinname, sequence and label. You can change the correnponding parameters in train_phosaf.py to complete your prediction tasks.

###Predict for your test data
If you want to predict your test data by the model, please prepare your test data as a csv file, one column is proteinname and the other column is sequence.
And then you can run the predict_phosaf.py to predict general or specific-kinase phoshphorylation sites by setting correnponding parameters.

##screen for reliable negative samples by secondary structure
The input file is a txt or fasta file, which includes negative proteinname and sequences.
Take a protein for example,'>sp|P53602|S|37 '\n'
                                           NIAVIKYWGKRDEELVLPINSSLSVTLHQDQLKTTTTAVIS'
And then you can run the screen_reliable_negsamples.py to screen negative samples on your data.

##Notes:
In file folder called feature, the information of three-dimensional coordinates of residues are getted from predicted structures by AlphaFold2, the information of angle and sasa of residues on each protein are getted from DSSP,  the information of pssm on proteins are obtained by running PSI-BLAST.
So if this folder does not contain the relevant features of proteins in your data, please use the tools above to obtain it.
