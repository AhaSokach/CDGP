import math
import pickle
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

def eval_popularity_prediction(model, criterion, data, n_neighbors,logger, device,
                               type="val", batch_size=200):
  val_loss = []
  all_pred = []
  all_target = []
  all_cas=[]
  pred_list = []
  with torch.no_grad():
    model = model.eval()
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      sources_batch = data.sources[s_idx:e_idx]
      destinations_batch = data.destinations[s_idx:e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]
      target_batch = data.labels[s_idx:e_idx]
      index = np.where(target_batch > 0)
      pred,return_fit = model.forward(sources_batch, destinations_batch, timestamps_batch,
                                            edge_idxs_batch, index, n_neighbors)
      if(sum(target_batch)>0):
        target_torch = torch.from_numpy(target_batch[index]).to(device)
        target = torch.log2(target_torch)
        pred = pred.to(torch.float32)
        target = target.to(torch.float32)
        loss = criterion(target, pred)
        val_loss.append(loss.item())
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        all_pred.extend(pred)
        all_target.extend(target)
        all_cas.extend(sources_batch[index])
    loss_test = sum(val_loss)/num_test_batch
    rmsle_test = rmsle(all_pred, all_target)
    msle_test = msle(all_pred, all_target)
    pcc_test = pcc(all_pred, all_target)
    male_test = male(all_pred, all_target)
    mape_test = mape(all_pred, all_target)
    logger.info(f"{type} loss:{loss_test}")
    logger.info(f"{type}  rmsle:{rmsle_test} msle:{msle_test} pcc:{pcc_test} male:{male_test} mape:{mape_test}")
    test_result = {"loss":loss_test, "rmsle":rmsle_test,
       "msle":msle_test, "pcc":pcc_test, "male":male_test,"mape":mape_test}
  return test_result


def rmsle(pred, label):
  return np.around(np.sqrt(mean_squared_error(label, pred)), 4)


def msle(pred, label):
  return np.around(mean_squared_error(label, pred), 4)


def pcc(pred, label):
  pred_mean, label_mean = np.mean(pred, axis=0), np.mean(label, axis=0)
  pre_std, label_std = np.std(pred, axis=0), np.std(label, axis=0)
  return np.around(np.mean((pred - pred_mean) * (label - label_mean) / (pre_std * label_std), axis=0), 4)


def male(pred, label):
  return np.around(mean_absolute_error(label, pred), 4)

def mape(pred, label):
  label = np.power(2,label)
  pred = np.power(2,pred)
  result = np.mean(np.abs(np.log2(pred + 1) - np.log2(label + 1)) / np.log2(label + 2))
  return np.around(result, 4)

