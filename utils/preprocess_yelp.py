import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime
from collections import defaultdict


def preprocess(data_name):
  user_counts = defaultdict(lambda: 0)
  biz_counts = defaultdict(lambda: 0)
  user_id_num = defaultdict(lambda: 0)
  biz_id_num = defaultdict(lambda: 0)
  links = []
  with open(data_name) as f:
    s = next(f)
    for idx, line in enumerate(f):
      e = json.loads(line)
      u_id = e["user_id"]
      b_id = e["business_id"]
      date = datetime.strptime(e["date"], '%Y-%m-%d %H:%M:%S')
      user_counts[u_id] += 1
      biz_counts[b_id] += 1
      user_id_num[u_id] = 0
      biz_id_num[b_id] = 0
      label = float(0)
      if idx % 10000 == 0:
        #break
        print(str(idx)+"...")
      links.append((idx, u_id, b_id, label, date))
  print(max(user_counts.values()))
  print(max(biz_counts.values()))
  print(max(user_id_num.values()))
  #print(links[0:5])
  links.sort(key = lambda x : x[4])
  #print(links[0:5])
  #print(links[-1])
  START_DATE = datetime.strptime("2009-01-01", '%Y-%m-%d')
  END_DATE = datetime.strptime("2016-11-01", '%Y-%m-%d')

  u_list, b_list, ts_list, label_list = [], [], [], []
  feat_l = []
  idx_list = []  # TODO idx not continuous
  u_s = 0
  b_s = 0
  idx_s = 0
  #print(biz_counts)
  for (idx, u, b, label, date) in links:

    if biz_counts[b] < 10:
      continue
    #print("##")
    if user_counts[u] < 10:
      continue
    #print("##")
    if date < START_DATE:
      continue
    if date > END_DATE:
      break

    ts = (date - START_DATE).total_seconds()

    # id to num
    if user_id_num[u] == 0:
      u_s += 1
      user_id_num[u] = u_s
      u = u_s
    else:
      u = user_id_num[u]
    if biz_id_num[b] == 0:
      b_s+=1
      biz_id_num[b] = b_s
      b = b_s
    else:
      b = biz_id_num[b]
    idx_s += 1
    idx = idx_s
    #feat = np.array([float(x) for x in e[4:]])  # TODO no edge feature
    feat = np.zeros(1)

    u_list.append(u)
    b_list.append(b)
    ts_list.append(ts)
    label_list.append(label)
    idx_list.append(idx)

    feat_l.append(feat)
  print("user node num:" + str(u_s))
  print("business node num:" + str(b_s))
  print("edge num:" + str(idx_s))
  print(ts_list[0])
  return pd.DataFrame({'u': u_list,
                       'i': b_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list}) , np.array(feat_l)


def reindex(df, bipartite=True):
  #df = df.sort_values(by="ts")
  #df.reset_index(drop=True)
  new_df = df.copy()
  print(min(new_df["ts"]))
  start_time = min(new_df["ts"])
  new_df["ts"]=new_df["ts"]-start_time  # start from zero
  print(min(new_df["ts"]))
  if bipartite:
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    upper_u = df.u.max()
    new_i = df.i + upper_u

    new_df.i = new_i
    #new_df.u += 1
    #new_df.i += 1
    #new_df.idx += 1
  else:
    new_df.u += 1
    new_df.i += 1
    #new_df.idx += 1

  return new_df


def run(bipartite=True):
  print("start...")
  data_name = "yelp_review.json"
  #Path("data/").mkdir(parents=True, exist_ok=True)
  PATH = './data/{}'.format(data_name)
  data_name = "yelp"
  OUT_DF = './data/ml_{}.csv'.format(data_name)
  OUT_FEAT = './data/ml_{}.npy'.format(data_name)
  OUT_NODE_FEAT = './data/ml_{}_node.npy'.format(data_name)

  df, feat = preprocess(PATH)
  new_df = reindex(df, bipartite)

  empty = np.zeros(feat.shape[1])[np.newaxis, :]
  feat = np.vstack([empty, feat])

  max_idx = max(new_df.u.max(), new_df.i.max())
  rand_feat = np.zeros((max_idx + 1, 172))

  new_df.to_csv(OUT_DF)
  np.save(OUT_FEAT, feat)
  np.save(OUT_NODE_FEAT, rand_feat)

if __name__ == '__main__':

  parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
##parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
#                    default='wikipedia')
#parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')

#args = parser.parse_args()
  print("run...")
  run(True)
#run(args.data, bipartite=args.bipartite)
