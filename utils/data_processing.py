import numpy as np
import random
import pandas as pd
import pickle


class Data:
  def __init__(self, sources, destinations, timestamps, edge_idxs, labels, types):
    self.sources = sources.astype(int)
    self.destinations = destinations.astype(int)
    self.timestamps = timestamps.astype(float)
    self.edge_idxs = edge_idxs.astype(int)
    self.labels = labels.astype(float)
    self.types = types.astype(int)
    self.n_interactions = len(sources)
    self.unique_nodes = set(sources) | set(destinations)


    self.n_unique_nodes = max(self.unique_nodes)+1
    print(self.n_unique_nodes)

def get_data_popularity(dataset_name):
  ### Load data and train val test split
  data = pickle.load(open('./data/{}/{}_withlabel.pkl'.format(dataset_name, dataset_name),"rb+"))
  node_features = np.load("./data/{}/{}_nodefeature.npy".format(dataset_name, dataset_name))
  edge_features = np.load("./data/{}/{}_edgefeature.npy".format(dataset_name, dataset_name))


  train_data = data["train"]
  val_data = data["val"]
  test_data = data["test"]

  '''min_time = train_data["time"].min()
  train_data["time"] = train_data["time"] - min_time
  val_data["time"] = val_data["time"] - min_time
  test_data["time"] = test_data["time"] - min_time'''

  all = pd.concat([train_data,val_data])
  all = pd.concat([all,test_data])
  all = all.to_numpy()
  all[:,0:3] = all[:,0:3].astype('int64')
  all[:, 5:6] = all[:, 5:6].astype(float)
  full_data = Data(all[:,1], all[:,0], all[:,2], all[:,3], all[:,5],all[:,6])

  train_data = train_data.to_numpy()
  train_data[:, 0:3] = train_data[:, 0:3].astype('int64')
  train_data[:, 5:6] = train_data[:, 5:6].astype(float)
  train_data = Data(train_data[:,1], train_data[:,0], train_data[:,2], train_data[:,3],
                     train_data[:,5],train_data[:,6])


  val_data = val_data.to_numpy()
  val_data[:, 0:3] = val_data[:, 0:3].astype('int64')
  val_data[:, 5:6] = val_data[:, 5:6].astype(float)
  val_data = Data(val_data[:,1], val_data[:,0], val_data[:,2],
                   val_data[:,3], val_data[:,5], val_data[:,6])

  test_data = test_data.to_numpy()
  test_data[:, 0:3] = test_data[:, 0:3].astype('int64')
  test_data[:, 5:6] = test_data[:, 5:6].astype(float)
  test_data = Data(test_data[:,1], test_data[:,0], test_data[:,2],
                    test_data[:,3], test_data[:,5],test_data[:,6])


  print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                               full_data.n_unique_nodes))

  return full_data, train_data, val_data, test_data ,node_features, edge_features ,full_data.n_unique_nodes, full_data.n_interactions


def compute_time_statistics(sources, destinations, timestamps):
  last_timestamp_sources = dict()
  last_timestamp_dst = dict()
  all_timediffs_src = []
  all_timediffs_dst = []
  for k in range(len(sources)):
    source_id = sources[k]
    dest_id = destinations[k]
    c_timestamp = timestamps[k]
    if source_id not in last_timestamp_sources.keys():
      last_timestamp_sources[source_id] = 0
    if dest_id not in last_timestamp_dst.keys():
      last_timestamp_dst[dest_id] = 0
    all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
    all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
    last_timestamp_sources[source_id] = c_timestamp
    last_timestamp_dst[dest_id] = c_timestamp
  assert len(all_timediffs_src) == len(sources)
  assert len(all_timediffs_dst) == len(sources)
  mean_time_shift_src = np.mean(all_timediffs_src)
  std_time_shift_src = np.std(all_timediffs_src)
  mean_time_shift_dst = np.mean(all_timediffs_dst)
  std_time_shift_dst = np.std(all_timediffs_dst)

  return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst






