import subprocess
import sys
import numpy as np

def return_ds_list(ds):
  ds_list=[]
  for i in range(len(ds['seq'])):
    ds_list.append((str(ds['sid'][i]),  str(ds['seq'][i])))
  return ds_list

def embedding(model_name,dataset,w):
	if model_name=='bert-base':
		#subprocess.check_call([sys.executable, '-m', 'pip', 'install','tape_proteins'])
		#!pip install tape_proteins
		import torch
		from tape import ProteinBertModel, TAPETokenizer
		model = ProteinBertModel.from_pretrained(model_name)
		tokenizer = TAPETokenizer(vocab='iupac') 
		X_ds=[]
		for i in(range(dataset.shape[0])):
			token_id = torch.tensor([tokenizer.encode(dataset['seq'][i])])
			output = model(token_id)[0]
			X_ds.append(output[0][1:2*w+2].detach().numpy())
		X_ds = np.dstack(X_ds)
		X_ds = np.rollaxis(X_ds,-1)
		return X_ds
	if model_name=='esm1v_t33_650M_UR90S_1':
		#subprocess.check_call([sys.executable, '-m', 'pip', 'install','pytorch-lightning'])
		#subprocess.check_call([sys.executable, '-m', 'pip', 'install','git+https://github.com/facebookresearch/esm.git'])
		#%pip install pytorch-lightning
		#%pip install git+https://github.com/facebookresearch/esm.git
		from esm import Alphabet, pretrained
		import os
		import torch
		from torch import nn
		import torch.nn.functional as F
		from torch.utils.data import DataLoader, random_split
		import pytorch_lightning as pl
		model, alphabet = pretrained.load_model_and_alphabet(model_name)
		batch_converter = alphabet.get_batch_converter()
		_,_,tokens=batch_converter(return_ds_list(dataset))

		batch_tokens = tokens[:, :w*2+3]
		with torch.no_grad():
			results = model(batch_tokens, repr_layers=[33])
		token_representations_tr = results['representations'][33]
		X_ds=token_representations_tr.cpu().detach().numpy()
		X_ds=X_ds[:,1:w*2+2,:]
		return X_ds
	if model_name.startswith('prot'):
		#subprocess.check_call([sys.executable, '-m', 'pip', 'install','transformers'])
		#subprocess.check_call([sys.executable, '-m', 'pip', 'install','SentencePiece'])
		#!pip install -q  transformers
		from transformers import TFBertModel, BertTokenizer, TFAlbertModel, AlbertTokenizer,TFXLNetModel, XLNetTokenizer
		if model_name.startswith('prot_bert'):
			tokenizer = BertTokenizer.from_pretrained("Rostlab/"+model_name, do_lower_case=False )
			model = TFBertModel.from_pretrained("Rostlab/"+model_name,from_pt=True)
			ids = tokenizer.batch_encode_plus(dataset['seq'], add_special_tokens=True, padding=True, return_tensors="tf")
			input_ids = ids['input_ids']
			attention_mask = ids['attention_mask']
			embedding = model(input_ids)[0]
			attention_mask = np.asarray(attention_mask)
			features = [] 
			for seq_num in range(len(embedding)):
				seq_len = (attention_mask[seq_num] == 1).sum()
				seq_emd = embedding[seq_num][1:seq_len-1]
				features.append(seq_emd)
			features = np.dstack(features)
			features = np.rollaxis(features,-1)
			return features
		if model_name=='prot_albert':
			tokenizer = AlbertTokenizer.from_pretrained("Rostlab/"+model_name, do_lower_case=False )
			model = TFAlbertModel.from_pretrained("Rostlab/"+model_name, from_pt=True)
			features = []
			shape1=dataset.shape[0] 
			strt=0
			next=500
			while strt<=shape1:
				if next>=shape1:
					next=shape1+1
				ids = tokenizer.batch_encode_plus(dataset['seq'][strt:next], add_special_tokens=True, padding=True, return_tensors="tf")
				input_ids = ids['input_ids']
				attention_mask = ids['attention_mask']
				embedding = model(input_ids)[0]
				attention_mask = np.asarray(attention_mask)
				for seq_num in range(len(embedding)):
					seq_len = (attention_mask[seq_num] == 1).sum()
					seq_emd = embedding[seq_num][1:seq_len-1]
					features.append(seq_emd)
				strt=next
				next=next+500
			features = np.dstack(features)
			features = np.rollaxis(features,-1)
			return features
		if model_name=='prot_xlnet':
			tokenizer = XLNetTokenizer.from_pretrained("Rostlab/"+model_name, do_lower_case=False )
			model = TFXLNetModel.from_pretrained("Rostlab/"+model_name,mem_len=512, from_pt=True)
			features = []
			shape1=dataset.shape[0] 
			strt=0
			next=1000
			while strt<=shape1:
				if next>=shape1:
					next=shape1+1
				ids = tokenizer.batch_encode_plus(dataset['seq'][strt:next], add_special_tokens=True, padding=True, return_tensors="tf")
				input_ids = ids['input_ids']
				attention_mask = ids['attention_mask']
				output = model(input_ids,attention_mask=attention_mask,mems=None)
				embedding = output.last_hidden_state
				memory = output.mems
				embedding = np.asarray(embedding)
				attention_mask = np.asarray(attention_mask)
				for seq_num in range(len(embedding)):
					seq_len = (attention_mask[seq_num] == 1).sum()
					padded_seq_len = len(attention_mask[seq_num])
					seq_emd = embedding[seq_num][padded_seq_len-seq_len:padded_seq_len-2]
					features.append(seq_emd)
				strt=next
				next=next+1000
			features = np.dstack(features)
			features = np.rollaxis(features,-1)
			return features
