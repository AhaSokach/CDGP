import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import datetime


def preprocess(data_name):
  u_list, i_list, ts_list, label_list = [], [], [], []
  feat_l = []
  idx_list = []
  links = []
  with open(data_name) as f:
    s = next(f)
    for idx, line in enumerate(f):
      e = line.strip().split(' ')
      if e[0] == "%":
        continue
      u = int(e[0])
      i = int(e[1])

      assert (int(e[2]) == 1)
      label = float(0)

      ts = float(e[3])  # TODO timestamp???

      timestamp = datetime.fromtimestamp(ts)
      links.append((u, i, label, ts))


  for (u, i, label, ts) in links:
    #feat = np.array([float(x) for x in e[4:]])  # TODO no edge feature
    feat = np.zeros(1)

    u_list.append(u)
    i_list.append(i)
    ts_list.append(ts)
    label_list.append(label)
    idx_list.append(idx)

    feat_l.append(feat)
  return pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list}) , np.array(feat_l)


def reindex(df, bipartite=True):
  df = df.sort_values(by="ts")
  df.reset_index(drop=True)
  new_df = df.copy()
  print(min(new_df["ts"]))
  start_time = min(new_df["ts"])
  new_df["ts"]=new_df["ts"]-start_time
  print(min(new_df["ts"]))
  if bipartite:
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    upper_u = df.u.max()
    new_i = df.i + upper_u

    new_df.i = new_i
    #new_df.u += 1
    #new_df.i += 1
    new_df.idx += 1
  else:
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

  return new_df


def run(bipartite=True):
  print("start...")
  data_name = "ia-movielens-user2tags-10m.edges"
  #Path("data/").mkdir(parents=True, exist_ok=True)
  PATH = './data/{}'.format(data_name)
  data_name = "movielens"
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
