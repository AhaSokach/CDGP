import pickle
import numpy as np
import random
import pandas as pd

def calculate_label(label,type):
    label_pop = np.array([])
    if type == "train" or type == "all":
        for dict in label:
            #label_pop.extend(dict["label_pop"])
            label_pop = np.concatenate((label_pop,dict["label_pop"]),axis=0)
    else:
        label_pop = label["label_pop"]
    min_label = min(label_pop)
    max_label = max(label_pop)
    avg_label = float(sum(label_pop)) / len(label_pop)
    print(f'{type}: min_label: {min_label}, max_label: {max_label}, '
          f'avg_label: {avg_label}, std_label: {label_pop.std()}')

def calculate_label_all(label):
    train = label["train"]
    val = label["val"]
    test = label["test"]
    label_pop = np.array([])
    for dict in train:
        # label_pop.extend(dict["label_pop"])
        label_pop = np.concatenate((label_pop, dict["label_pop"]), axis=0)
    label_pop = np.concatenate((label_pop, val["label_pop"],test["label_pop"]), axis=0)
    min_label = min(label_pop)
    max_label = max(label_pop)
    avg_label = float(sum(label_pop)) / len(label_pop)
    print(f'{type}: min_label: {min_label}, max_label: {max_label}, '
          f'avg_label: {avg_label}, std_label: {label_pop.std()}')


def print_data(list,type):
    list=np.array(list)
    print(f'{type}: min: {min(list)}, max: {max(list)}, '
          f'avg: {sum(list)/len(list)}, std: {list.std()}')

if __name__ == '__main__':

    #result = pickle.load(open('/home/luxd/popularity/dctgn/results/tw-com-nodiff-new.pkl',"rb+"))
    #print(result)
    #community = result["test_community_index"].cpu().numpy()

    dataset_name = "twitter"
    label = pickle.load(open('../data/{}/{}_label.pkl'.format(dataset_name, dataset_name), "rb+"))

    data = pickle.load(open('../data/{}/{}_withlabel.pkl'.format(dataset_name, dataset_name),"rb+"))
    #data = pd.read_csv('../data/{}/{}.csv'.format(dataset_name, dataset_name))
    data = pd.concat([data["train"],data["val"],data["test"]])
    calculate_label(label["train"],"train")
    calculate_label(label["val"],"val")
    calculate_label(label["test"],"test")
    #calculate_label_all(label)

    casgroup = data.groupby(by="cas")
    cas_len = []
    cas_train_len=[]
    cas_val_len=[]
    cas_test_len=[]
    cas_label_minus=[]
    cas_label_minus_p = []
    community_cas_len = []
    cas_num = 0
    dst_num = 0
    for cas, group in casgroup:
        cas_num+=1
        group.sort_values(by="time", inplace=True, ascending=True)
        #index = community[community == cas]
        cas_len.append(group.shape[0])
        #if (len(index) != 0):
        #    community_cas_len.append(group.shape[0])
        group_train = group[group["type"]==1]
        group_val = group[group["type"]==2]
        group_test = group[group["type"] == 3]
        cas_train_len.append(group_train.shape[0])
        cas_val_len.append(group_val.shape[0])
        cas_test_len.append(group_test.shape[0])
        label_test = group_test[group_test["label"] != 0]
        if(label_test.shape[0]!=0):
            pop = label_test["label"].tolist()[0]
            label_train = group_train[group_train["label"] != 0]["label"].tolist()
            cas_label_minus.extend([abs(i - pop) for i in label_train])
            cas_label_minus_p.extend([float(abs(i - pop))/pop for i in label_train])




    print_data(cas_train_len,"cas_train_len")
    print_data(cas_val_len,"cas_val_len")
    print_data(cas_test_len,"cas_test_len")
    print_data(cas_label_minus,"test_train_minus")
    print_data(cas_label_minus_p,"test_train_minus_percent")

    print(f'max cas len{max(cas_len)}')
    print(f'min cas len{min(cas_len)}')
    #print(f'max community cas len{max(community_cas_len)}')
    print(f'avg cas len{float(sum(cas_len))/len(cas_len)}')
    #print(f'avg community cas len{float(sum(community_cas_len))/len(community_cas_len)}')
    #print(f'community cas len{len(community_cas_len)}')
    #print(community_cas_len)

    casgroup = data.groupby(by="dst")
    dst_len = []
    community_dst_len = []
    for dst, group in casgroup:
        dst_num+=1
        group.sort_values(by="time", inplace=True, ascending=True)
        #index = community[community == dst]
        dst_len.append(group.shape[0])
        #if (len(index) != 0):
        #    community_dst_len.append(group.shape[0])

    print(f'max dst len{max(dst_len)}')
    print(f'min dst len{min(dst_len)}')

    #print(f'max community dst len{max(community_dst_len)}')

    print(f'avg dst len{float(sum(dst_len)) / len(dst_len)}')
    #print(f'avg community dst len{float(sum(community_dst_len)) / len(community_dst_len)}')
    #print(f'community dst len{len(community_dst_len)}')

    print(f'dst num:{dst_num} cas num:{cas_num} ')
