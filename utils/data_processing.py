import numpy as np
import random
import pandas as pd
import pickle
import json

class Data:
  def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
    self.sources = sources
    self.destinations = destinations
    self.timestamps = timestamps
    self.edge_idxs = edge_idxs
    self.labels = labels
    #self.types = types
    self.n_interactions = len(sources)
    self.unique_nodes = set(sources) | set(destinations)
    self.n_unique_nodes = max(self.unique_nodes) + 1
    print(self.n_unique_nodes)

def set_type(self, types):
    self.types = types

class Data2:
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


def get_data_node_classification(dataset_name, use_validation=False):
  ### Load data and train val test split
  graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
  edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
  node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name))

  val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values
  timestamps = graph_df.ts.values

  random.seed(2020)

  train_mask = timestamps <= val_time if use_validation else timestamps <= test_time
  test_mask = timestamps > test_time
  val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time) if use_validation else test_mask

  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])

  val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask])

  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask])

  return full_data, node_features, edge_features, train_data, val_data, test_data


def get_data_popularity(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False):
  ### Load data and train val test split
  data = pickle.load(open('./data/{}/{}_withlabel.pkl'.format(dataset_name, dataset_name),"rb+"))
  #graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
  node_features = np.load("./data/{}/{}_nodefeature.npy".format(dataset_name, dataset_name))
  edge_features = np.load("./data/{}/{}_edgefeature.npy".format(dataset_name, dataset_name))

  #if randomize_features:
  #  node_features = np.random.rand(node_features.shape[0], node_features.shape[1])

  #val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

  train_data = data["train"]
  val_data = data["val"]
  test_data = data["test"]
  all = pd.concat([train_data,val_data])
  all = pd.concat([all,test_data])
  all = all.to_numpy()
  all[:,0:3] = all[:,0:3].astype(int)
  all[:, 5:6] = all[:, 5:6].astype(float)
  full_data = Data2(all[:,1], all[:,0], all[:,2], all[:,3], all[:,5],all[:,6])

  train_data = train_data.to_numpy()
  train_data[:, 0:3] = train_data[:, 0:3].astype(int)
  train_data[:, 5:6] = train_data[:, 5:6].astype(float)
  train_data = Data2(train_data[:,1], train_data[:,0], train_data[:,2], train_data[:,3],
                     train_data[:,5],train_data[:,6])


  val_data = val_data.to_numpy()
  val_data[:, 0:3] = val_data[:, 0:3].astype(int)
  val_data[:, 5:6] = val_data[:, 5:6].astype(float)
  val_data = Data2(val_data[:,1], val_data[:,0], val_data[:,2],
                   val_data[:,3], val_data[:,5], val_data[:,6])

  test_data = test_data.to_numpy()
  test_data[:, 0:3] = test_data[:, 0:3].astype(int)
  test_data[:, 5:6] = test_data[:, 5:6].astype(float)
  test_data = Data2(test_data[:,1], test_data[:,0], test_data[:,2],
                    test_data[:,3], test_data[:,5],test_data[:,6])


  random.seed(2022)

  print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                               full_data.n_unique_nodes))

  return full_data, train_data, val_data, test_data ,node_features, edge_features ,full_data.n_unique_nodes, full_data.n_interactions


def get_data(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False):
  ### Load data and train val test split
  graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
  edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
  node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name)) 
    
  if randomize_features:
    node_features = np.random.rand(node_features.shape[0], node_features.shape[1])

  val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values
  timestamps = graph_df.ts.values

  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

  random.seed(2020)

  node_set = set(sources) | set(destinations) #union set
  n_total_unique_nodes = len(node_set)

  # Compute nodes which appear at test time
  test_node_set = set(sources[timestamps > val_time]).union(
    set(destinations[timestamps > val_time]))
  # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
  # their edges from training
  new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))

  # Mask saying for each source and destination whether they are new test nodes
  new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
  new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

  # Mask which is true for edges with both destination and source not being new test nodes (because
  # we want to remove all edges involving any new test node)
  observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

  # For train we keep edges happening before the validation time which do not involve any new node
  # used for inductiveness
  train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)

  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])

  # define the new nodes sets for testing inductiveness of the model
  train_node_set = set(train_data.sources).union(train_data.destinations)
  assert len(train_node_set & new_test_node_set) == 0
  new_node_set = node_set - train_node_set

  val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
  test_mask = timestamps > test_time

  if different_new_nodes_between_val_and_test:
    n_new_nodes = len(new_test_node_set) // 2
    val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
    test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])

    edge_contains_new_val_node_mask = np.array(
      [(a in val_new_node_set or b in val_new_node_set) for a, b in zip(sources, destinations)])
    edge_contains_new_test_node_mask = np.array(
      [(a in test_new_node_set or b in test_new_node_set) for a, b in zip(sources, destinations)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)


  else:
    edge_contains_new_node_mask = np.array(
      [(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

  # validation and test with all edges
  val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask])

  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask])

  # validation and test with edges that at least has one new node (not in training set)
  new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                           timestamps[new_node_val_mask],
                           edge_idxs[new_node_val_mask], labels[new_node_val_mask])

  new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                            timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                            labels[new_node_test_mask])

  print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                      full_data.n_unique_nodes))
  print("The training dataset has {} interactions, involving {} different nodes".format(
    train_data.n_interactions, train_data.n_unique_nodes))
  print("The validation dataset has {} interactions, involving {} different nodes".format(
    val_data.n_interactions, val_data.n_unique_nodes))
  print("The test dataset has {} interactions, involving {} different nodes".format(
    test_data.n_interactions, test_data.n_unique_nodes))
  print("The new node validation dataset has {} interactions, involving {} different nodes".format(
    new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
  print("The new node test dataset has {} interactions, involving {} different nodes".format(
    new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
  print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(
    len(new_test_node_set)))

  return node_features, edge_features, full_data, train_data, val_data, test_data, \
         new_node_val_data, new_node_test_data


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



if __name__ == '__main__':
  dataset_name = "aminer"
  label = pickle.load(open('../data/{}2/{}2_label.pkl'.format(dataset_name, dataset_name),"rb+"))
  data = pd.read_csv('../data/{}2/{}2.csv'.format(dataset_name, dataset_name))
  #with open('../data/preprocess/config/config.json', 'r') as f:
  #  param = json.load(f)
  param=label
  data.insert(loc=4,column="label",value=0)
  data.insert(loc=5,column="type",value=0)
  casgroup = data.groupby(by="cas")
  train_label = label["train"]
  val_label = label["val"]
  test_label = label["test"]
  max_id = max(data["dst"]) # reindex
  df_train = pd.DataFrame(columns=["dst","cas","time","eid","src","label","type"])
  df_val = pd.DataFrame(columns=["dst","cas","time","eid","src","label","type"])
  df_test = pd.DataFrame(columns=["dst","cas","time","eid","src","label","type"])
  max_e_train = max(train_label[-1]["eid"])
  max_e_val = max(val_label["eid"])
  max_e_test = max(test_label["eid"])
  edge_feature = np.zeros(data.shape[0])
  split_train_time = param["train_time"][-1]
  split_val_time = param["val_time"]
  split_test_time = param["test_time"]
  for cas,group in casgroup:
    size = group.shape[0]
    group.sort_values(by="time", inplace=True, ascending=True)
    group['m_time'] = pd.to_datetime(data["time"],unit='s')
    max_id += 1
    #if dataset_name != "aminer":
    group["cas"] = max_id
    eid_list=[]
    label_list=[]
    type_list=[]
    for dict in train_label:
      id_list = dict["id"]
      index = np.where(id_list == cas)
      if(len(index[0])!=0):
        eid = dict["eid"]
        label_pop=dict["label_pop"]
        eid_list.extend(list(eid[index[0]]))
        label_list.extend(list(label_pop[index[0]]))
        type_list.extend([1]*len(index[0]))
    i=0
    index_train = 0
    e_train=0
    for e in eid_list:
      index2= group["eid"] == e
      group.loc[index2,"label"]=label_list[i]
      i+=1
    index_train=group["m_time"]<=split_train_time
    group.loc[index_train,"type"]=1
    df_train = pd.concat([df_train, group[index_train]])
    size3 = group[index_train].shape[0]


    #index3 = data["cas"] == cas
    #data["cas"].replace(to_replace=cas,value=max_id,inplace=True)
    id_list = val_label["id"]
    index = np.where(id_list == cas)
    index = index[0]
    assert len(index) <= 1

    e_val = e_train
    if(len(index)!=0):
      eid = list(val_label["eid"][index])[0]
      label_pop=list(val_label["label_pop"][index])[0]
      #eid_list.extend(list(eid[index]))
      #label_list.extend(list(label_pop[index]))
      #type_list.extend([2])
      index3 = group["eid"] == eid
      group.loc[index3, "label"] = label_pop

    index_val = ((group["m_time"] > split_train_time) & (group["m_time"] <= split_val_time))
 
    group.loc[index_val, "type"] = 2
    df_val = pd.concat([df_val, group[index_val]])
    size3 += group[index_val].shape[0]

    id_list = test_label["id"]
    index = np.where(id_list == cas)
    index = index[0]
    assert len(index) <= 1

    e_test = e_val
    if (len(index) != 0):
      eid = list(test_label["eid"][index])[0]
      label_pop = list(test_label["label_pop"][index])[0]
      # eid_list.extend(list(eid[index]))
      # label_list.extend(list(label_pop[index]))
      # type_list.extend([2])
      index3 = group["eid"] == eid
      group.loc[index3, "label"] = label_pop
    index_test = ((group["m_time"] > split_val_time) & (group["m_time"] <= split_test_time))
    group.loc[index_test, "type"] = 3
    df_test = pd.concat([df_test, group[index_test]])
    size3 += group[index_test].shape[0]

    test_test = group[group["m_time"] > split_test_time]
    size2 = size3 + test_test.shape[0]
    if (size != size2):
      print(cas)
      print(size - size2)
      print("train")
      print(df_train[df_train["cas"] == cas]["eid"].tolist())
      print("Val")
      print(df_val[df_val["cas"] == cas]["eid"].tolist())
      print("test")
      print(df_test[df_test["cas"] == cas]["eid"].tolist())
      print(group[group["cas"] == cas]["eid"].tolist())





  df_train.drop(labels='m_time', inplace=True, axis=1)
  df_val.drop(labels='m_time', inplace=True, axis=1)
  df_test.drop(labels='m_time', inplace=True, axis=1)

  df_train.sort_values(by="time", inplace=True, ascending=True)
  df_val.sort_values(by="time", inplace=True, ascending=True)
  df_test.sort_values(by="time", inplace=True, ascending=True)


  f = open("../data/{}2/{}2_withlabel.pkl".format(dataset_name, dataset_name),'wb')

  datas = {
     'train':df_train,'val':df_val,'test':df_test}

  data= pickle.dump(datas,f,-1)

  #f.close()
  max_id+=1
  node_feature = np.zeros((max_id,172))

  np.save("../data/{}2/{}2_nodefeature.npy".format(dataset_name, dataset_name),node_feature)
  np.save("../data/{}2/{}2_edgefeature.npy".format(dataset_name, dataset_name),edge_feature)
  print("over")






