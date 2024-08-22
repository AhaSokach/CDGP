import numpy as np
import random
import pandas as pd
import pickle
import json

if __name__ == '__main__':
  dataset_name = "twitter"
  label = pickle.load(open('/home/jishuo/dataset/dataset-2/{}/{}_label.pkl'.format(dataset_name, dataset_name),"rb+"))
  data = pd.read_csv('/home/jishuo/dataset/dataset-2/{}/{}.csv'.format(dataset_name, dataset_name))
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

    id_list = val_label["id"]
    index = np.where(id_list == cas)
    index = index[0]
    assert len(index) <= 1

    e_val = e_train
    if(len(index)!=0):
      eid = list(val_label["eid"][index])[0]
      label_pop=list(val_label["label_pop"][index])[0]
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



  df_train.drop(labels='m_time', inplace=True, axis=1)
  df_val.drop(labels='m_time', inplace=True, axis=1)
  df_test.drop(labels='m_time', inplace=True, axis=1)

  df_train.sort_values(by="time", inplace=True, ascending=True)
  df_val.sort_values(by="time", inplace=True, ascending=True)
  df_test.sort_values(by="time", inplace=True, ascending=True)


  f = open("../data/{}/{}_withlabel.pkl".format(dataset_name, dataset_name),'wb')

  datas = {
     'train':df_train,'val':df_val,'test':df_test}

  data= pickle.dump(datas,f,-1)

  #f.close()
  max_id+=1
  node_feature = np.zeros((max_id,172))

  np.save("../data/{}/{}_nodefeature.npy".format(dataset_name, dataset_name),node_feature)
  np.save("../data/{}/{}_edgefeature.npy".format(dataset_name, dataset_name),edge_feature)
  print("over")
