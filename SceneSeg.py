import os
import pickle
import json
import numpy as np
import collections
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from evaluate_sceneseg import calc_ap, calc_miou


'''
#####
prediction: probability of shot boundary being a scene boundary
64 movies in total
gt: i-th element corresponds to the transition between the i-th and the (i+1)-th shots.

k		v
imdb_id "tt2488496"
place	2396 * 2048 (2396 as number of shots)
cast	2396 * 512
action	2396 * 512
audio	2396 * 512
scene_transition_boundary_ground_truth	1 * 2396 (True/False) # evaluation only
shot_end_frame	1 * 2396 # evaluation only
scene_transition_boundary_prediction	1 * 2395

#####
Metric: 
calc_ap(): AP, mAP, AP_dict
calc_miou(): mean_miou, miou_dict

#####
https://anyirao.com/projects/SceneSeg.html
'''



def data_peek():
	data_path = "./data"
	for f in os.listdir(data_path):
		if f == "tt2488496.pkl":
			data = pickle.load(open(os.path.join(data_path, f), "rb"))
	for k, v in data.items():
		print (k, v)


def data_preprocess():
	# preprocess data for further modeling data preparation
	data_path = "./data"
	files = []
	for f in os.listdir(data_path):
		files.append(os.path.join(data_path, f))

	data_dict = dict()
	for f_path in files: # formatting data
		movie_dict = pickle.load(open(f_path, "rb"))
		imdb_id = movie_dict["imdb_id"]
		place = movie_dict["place"]
		cast = movie_dict["cast"]
		action = movie_dict["action"]
		audio = movie_dict["audio"]
		scene_transition_boundary_gt = movie_dict["scene_transition_boundary_ground_truth"]
		shot_end_frame = movie_dict["shot_end_frame"]
		# data_dict[imdb_id] = {"place": place,
		# 					   "cast": cast,
		# 					   "action": action,
		# 					   "audio": audio,
		# 					   "shot_idx": shot_end_frame,
		# 					   "scene_gt": scene_transition_boundary_ground_truth} # ground truth
		shot_end_frame = shot_end_frame.view(shot_end_frame.shape[0], 1) # make it 1 * 2178
		scene_transition_boundary_gt = scene_transition_boundary_gt.view(scene_transition_boundary_gt.shape[0], 1)
		data_dict[imdb_id] = [torch.cat((place, cast, action, audio), 1), \
						   shot_end_frame, scene_transition_boundary_gt]
		# 3584 = 2048 + 512 * 3
		# input: 2178 * 3584 => output: 2178 + 2177 (shot_idx + gt)
		# data_array.append([torch.cat((place, cast, action, audio), 1), \
		# 				   torch.cat((shot_end_frame, scene_transition_boundary_ground_truth), 1)])
	pickle.dump(data_dict, open("data_dict.p", "wb"))
	return data_dict


class SceneSegDataset(Dataset):
	# dedicated dataset for dataloader
	def __init__(self, features, shot_idx, scene_gt, imdbids):
		self.features = features
		self.shot_idx = shot_idx
		self.scene_gt = scene_gt
		self.imdbids = imdbids

	def __len__(self):
		return len(self.features)

	def __getitem__(self, idx):
		'Generates one random sample of data'
		f_tensor = self.features[idx] # 2178 * 3548
		s_tensor = self.shot_idx[idx] # 2178 * 1
		gt_tensor = self.scene_gt[idx] # 2177 * 1
		imdbid = self.imdbids[idx]
		return f_tensor, s_tensor, gt_tensor, imdbid


def train_test_prepare(data_array, batch_size):
	# setting data splits and create dataloader for training and test
	imdbids = list(data_array.keys())
	np.random.shuffle(imdbids)
	num_train = int(len(imdbids) * 0.8) # movies for train
	train_idxs = imdbids[:num_train]
	train_feats, train_shot_idx = [], []
	train_scene_gt, train_movie_ids = [], []
	for imdbid in train_idxs:
		shots_num = len(data_array[imdbid][0]) # shots
		for i in range(shots_num):
			train_feats.append(data_array[imdbid][0][i])
			train_shot_idx.append(data_array[imdbid][1][i])
			try: # for last shot
				train_scene_gt.append(data_array[imdbid][2][i])
			except:
				train_scene_gt.append(torch.tensor([0]))
			train_movie_ids.append(imdbid)
	train_data = SceneSegDataset(train_feats, train_shot_idx, train_scene_gt, train_movie_ids)
	train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

	test_idxs = imdbids[num_train:]
	test_feats, test_shot_idx = [], []
	test_scene_gt, test_movie_ids = [], []
	for imdbid in test_idxs:
		shots_num = len(data_array[imdbid][0]) # shots
		for i in range(shots_num):
			test_feats.append(data_array[imdbid][0][i])
			test_shot_idx.append(data_array[imdbid][1][i])
			try:
				test_scene_gt.append(data_array[imdbid][2][i])
			except:
				test_scene_gt.append(torch.tensor([0]))
			test_movie_ids.append(imdbid)
	test_data = SceneSegDataset(test_feats, test_shot_idx, test_scene_gt, test_movie_ids)
	test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
	return train_loader, test_loader


class MyModel(nn.Module):
	# naive 2 layers of FC for binary classficiation
	def __init__(self, input_size, hidden_size, batch_size):
		super(MyModel, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.fc1 = nn.Linear(self.input_size, self.hidden_size)
		self.fc2 = nn.Linear(self.hidden_size, 1)

	def forward(self, x):
		output1 = self.fc1(x)
		out = torch.sigmoid(self.fc2(output1)) # .squeeze()
		return out


def set_random_seed(seed):
	# random seed setting
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def train(train_loader, test_loader, input_size, hidden_size, epochs, batch_size, lr, data_array):
	device = torch.device("cpu")
	set_random_seed(1)
	model = MyModel(input_size, hidden_size, batch_size).to(device)
	criterion = nn.BCELoss(reduction='none')
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	# train
	model.train()
	step = 0
	for epoch in range(epochs):
		for feats, shot_idx, scene_gt, imdbid in train_loader:
			scene_gt = scene_gt.float()
			# print (feats, "xxxxx", shot_idx, "xxxxx", scene_gt)
			model.zero_grad()
			pred = model(feats)
			# print (pred.shape, scene_gt.shape)
			loss = criterion(pred, scene_gt).mean()
			loss.backward()
			optimizer.step()
			step += 1
			if step % 100 == 0:
				print (f"Epoch: {epoch}, loss: {loss}")

	# test
	model.eval()
	pred_dict_tmp = collections.defaultdict(list)
	for feats, shot_idx, _, imdbid in test_loader:
		p = model(feats)
		y = torch.where(p > 0.5, torch.ones_like(p), torch.zeros_like(p))
		print (len(list(set(imdbid))))
		imdbid = imdbid[0]
		pred_dict_tmp[imdbid].append([shot_idx, y])


	pickle.dump(pred_dict_tmp, open("pred_dict_raw.p", "wb"))
	pr_dict, gt_dict, shot_to_end_frame_dict = {}, {}, {}
	for imdbid, tp in pred_dict_tmp.items():
		shot_idx = data_array[imdbid][1] # shot idx
		gt = data_array[imdbid][2] # gt
		shots_num = len(shot_idx)
		pred_vals = [0] * (shots_num-1)
		idx = 0
		for _, y in tp[:-1]:
			pred_vals[idx] = int(y)
			idx += 1

		pred_vals = torch.tensor(pred_vals).view(len(pred_vals), )
		pr_dict[imdbid] = pred_vals
		gt_dict[imdbid] = gt.view(len(gt), )
		shot_to_end_frame_dict[imdbid] = shot_idx.view(len(shot_idx), )
		shot_to_end_frame_dict = shot_idx

	res = [pr_dict, gt_dict, shot_to_end_frame_dict]
	pickle.dump(res, open("res.p", "wb"))
	_, mAP, _ = calc_ap(gt_dict, pr_dict)
	print (f"mAP: {mAP}")
	mMiou, _ = calc_miou(gt_dict, pr_dict, shot_to_end_frame_dict, threshold=0.5)
	print (f"mMiou: {mMiou}")


def main():
	# model hyperparameters
	lr = 1e-3
	input_size = 2048 + 512 + 512 + 512 # 3584
	hidden_size = 512 # hidden
	batch_size = 1
	epochs = 5

	# data separate
	data_array = data_preprocess()
	# data_array = pickle.load(open("data_dict.p", "rb"))
	train_loader, test_loader = train_test_prepare(data_array, batch_size)
	# model train
	train(train_loader, test_loader, input_size, hidden_size, epochs, batch_size, lr, data_array)


def evaluate():
	# evaluate from the saved trained results
	data_array = pickle.load(open("data_dict.p", "rb"))
	pred_dict_tmp = pickle.load(open("pred_dict_raw.p", "rb"))
	pr_dict, gt_dict, shot_to_end_frame_dict = {}, {}, {}
	for imdbid, tp in pred_dict_tmp.items():
		shot_idx = data_array[imdbid][1] # shot idx
		gt = data_array[imdbid][2] # gt
		shots_num = len(shot_idx)
		pred_vals = [0] * (shots_num-1)
		idx = 0
		for _, y in tp[:-1]:
			pred_vals[idx] = int(y)
			idx += 1

		pred_vals = torch.tensor(pred_vals).view(len(pred_vals), )
		pr_dict[imdbid] = pred_vals
		gt_dict[imdbid] = gt.view(len(gt), )
		shot_to_end_frame_dict[imdbid] = shot_idx.view(len(shot_idx), )

	_, mAP, _ = calc_ap(gt_dict, pr_dict)
	print (f"mAP: {mAP}") # 0.09946424285097273
	mMiou, _ = calc_miou(gt_dict, pr_dict, shot_to_end_frame_dict, threshold=0.5)
	print (f"mMiou: {mMiou}") # 0.03892633361808894


if __name__ == "__main__":
	# data_peek()
	main()
	# evaluate()















