import math
import pickle
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, average_precision_score, roc_auc_score


def eval_popularity_prediction(model, criterion, data,logger, device,
                               type="val", batch_size=100):
  val_loss = []
  all_pred = []
  all_target = []
  all_cas=[]
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
      pred = model.forward(sources_batch, destinations_batch, timestamps_batch,
                                            edge_idxs_batch, index)
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
  return test_result,all_pred,all_target,all_cas


def eval_edge_prediction(model, negative_edge_sampler, data, batch_size=100):
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_ap, val_auc = [], []
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
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

      size = len(sources_batch)
      _, negative_samples = negative_edge_sampler.sample(size)

      pos_prob, neg_prob = model.forward_edge(sources_batch, destinations_batch,
                                                            negative_samples, timestamps_batch,
                                                            edge_idxs_batch, )

      pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
      true_label = np.concatenate([np.ones(size), np.zeros(size)])

      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))

  return np.mean(val_ap), np.mean(val_auc)

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

