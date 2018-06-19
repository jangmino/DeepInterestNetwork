import os
import time
import pickle
import random
import numpy as np
import tensorflow as tf
import sys
import argparse
import csv

from input import DataInput, DataInputTest
from model import Model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

test_batch_size = 512

# use wepick_dataset.pkl
with open('wepick_dataset.pkl', 'rb') as f:
  train_set = pickle.load(f)
  test_set = pickle.load(f)
  cate_list = pickle.load(f)
  user_count, item_count, cate_count = pickle.load(f)

best_auc = 0.0
def calc_auc(raw_arr):
    """Summary

    Args:
        raw_arr (TYPE): Description

    Returns:
        TYPE: Description
    """
    # sort by pred value, from small to big
    arr = sorted(raw_arr, key=lambda d:d[2])

    auc = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    for record in arr:
        fp2 += record[0] # noclick
        tp2 += record[1] # click
        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2

    # if all nonclick or click, disgard
    threshold = len(arr) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        return -0.5

    if tp2 * fp2 > 0.0:  # normal auc
        return (1.0 - auc / (2.0 * tp2 * fp2))
    else:
        return None

def _auc_arr(score):
  score_p = score[:,0]
  score_n = score[:,1]
  #print "============== p ============="
  #print score_p
  #print "============== n ============="
  #print score_n
  score_arr = []
  for s in score_p.tolist():
    score_arr.append([0, 1, s])
  for s in score_n.tolist():
    score_arr.append([1, 0, s])
  return score_arr
def _eval(sess, model, model_path):
  auc_sum = 0.0
  score_arr = []
  total_time = 0
  total_model = 0
  start = time.time()
  for _, uij in DataInputTest(test_set, test_batch_size):
    inf_start = time.time()
    auc_, score_ = model.eval(sess, uij)
    inf_end = time.time()
    total_model += inf_end - inf_start
    score_arr += _auc_arr(score_)
    auc_sum += auc_ * len(uij[0])
  test_gauc = auc_sum / len(test_set)
  total_time += (time.time() - start)
  #sys.stderr.write("Elapsed total {}: model {}\n".format(total_time, total_model))
  sys.stderr.flush()

  Auc = calc_auc(score_arr)
  global best_auc
  if best_auc < test_gauc:
    best_auc = test_gauc
    model.save(sess, model_path)
  return test_gauc, Auc

def build_i_map(keys):
  """
  Make inverse map for keys: i -> item
  :param keys: listing items in order.
  :return:
  """
  return dict(zip(range(len(keys), keys)))

def restore_info(uij, predicted, dic):
  iu, ii, _, ihist_i, _ = uij

  u = dic['deal_key'][iu]
  for i in range(len(iu[0])):
    u = dic['deal_key'][iu[i]]

def get_wepick_setting(data_path=r'd:\WMIND\temp\wepick_settings.txt'):
  '''
  위픽 세팅 데이터
  :param data_path:
  :return:
  '''
  with open(data_path) as csvfile:
    reader = csv.reader(csvfile)
    wepick_setting_dic = {}
    for row in reader:
      deal_info = row[0].strip().split(' ')
      cnt, did = int(deal_info[0]), int(deal_info[1])
      slot = int(row[1])
      dt = row[2]
      if dt not in wepick_setting_dic:
        wepick_setting_dic[dt] = {}
      if slot not in wepick_setting_dic[dt] or cnt > wepick_setting_dic[dt][slot][1]:
        wepick_setting_dic[dt][slot] = (did, cnt)
    return wepick_setting_dic

def inspect_item_b(sess, model, out_file='./item_b.csv'):
  with open('wepick_data.pkl', 'rb') as f:
    wepick_data = pickle.load(f)

  dt = '2018-04-11 21'
  wepick_setting_dic = get_wepick_setting()
  _dids = []
  deal2slot = {}
  for slot, deal_info in wepick_setting_dic[dt].items():
    if slot >= FLAGS.predict_slot_after:
      _dids.append(deal_info[0])
      deal2slot[deal_info[0]] = slot
  effective_dids = set(_dids)

  item_b = model.prob_item_b(sess)
  sort_i = np.argsort(item_b)
  sort_i = np.fliplr([sort_i])[0]

  item_b_dic = list(zip(list(map(lambda x: wepick_data['deal_key'][x], sort_i)), item_b[sort_i]))

  with open(out_file, 'w') as item_b_f:
    for i,s in item_b_dic:
      if i in deal2slot:
        item_b_f.write("{},{},{}\n".format(i, deal2slot[i], s))

def _predict(sess, model, out_file):
  with open('wepick_data.pkl', 'rb') as f:
    wepick_data = pickle.load(f)
    #deal_i_map = build_i_map(wepick_data['deal_map'])

  dt = '2018-04-11 21'
  wepick_setting_dic = get_wepick_setting()
  _dids = []
  deal2slot = {}
  for slot, deal_info in wepick_setting_dic[dt].items():
    if slot >= FLAGS.predict_slot_after:
      _dids.append(deal_info[0])
      deal2slot[deal_info[0]] = slot
  effective_dids = set(_dids)

  total_model = 0
  total_time = 0
  start = time.time()

  with open(out_file, 'w') as pred_f:
    for _, uij in DataInputTest(test_set, FLAGS.predict_batch_size):
      outputs = []
      inf_start = time.time()
      users, histories, lengths = uij[0], uij[3], uij[4]
      predicted = model.test(sess, users, histories, lengths)
      inf_end = time.time()
      total_model += inf_end - inf_start

      # x: 간접 딜인덱스
      # wepick_data['deal_key']: 직접 딜번호
      for k in range(len(users)):
        u = wepick_data['user_key'][users[k]]
        hist = list(map(lambda x: wepick_data['deal_key'][x], histories[k][0:lengths[k]]))
        sort_i = np.argsort(predicted[k,:])
        sort_i = np.fliplr([sort_i])[0]
        order = list(filter(lambda x: x[0] in effective_dids,
          map(lambda x: (wepick_data['deal_key'][x],
                         deal2slot[wepick_data['deal_key'][x]] if wepick_data['deal_key'][x] in deal2slot else None,
                         predicted[k, x]), sort_i))
        )
        outputs.append((u, hist, order))

      for u, hist, order in outputs:
        h = "-".join(map(lambda x: str(x), hist))
        s = ":".join(map(lambda x: "{}/{}/{:.2f}".format(x[0], x[1], x[2]), order))
        pred_f.write("{},{},{}\n".format(u, h, s))
    #

  total_time += (time.time() - start)
  sys.stderr.write("Elapsed total {}: model {}\n".format(total_time, total_model))

def _make_suffix(arch_dict):
  return "a{}_{}_{}_e{}".format(arch_dict['emb_dim'],
                                arch_dict['f1_dim'],
                                arch_dict['f2_dim'],
                                ('T' if FLAGS.use_item_embedding else 'F')
                                )

def main(_):
  gpu_options = tf.GPUOptions(allow_growth=True)
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    s = FLAGS.architecture.split('_')
    arch_dict = {'emb_dim':int(s[0]), 'f1_dim':int(s[1]), 'f2_dim':int(s[2])}
    path_suffix = _make_suffix(arch_dict)
    model_prefix = os.path.join("save_path", "save_{}".format(path_suffix))
    model_path = os.path.join(model_prefix, "ckpt")
    log_dir = os.path.join("log", "log_{}".format(path_suffix))
    pred_dir = "prediction"
    item_b_pred_file = os.path.join(pred_dir, "item_b_pred_{}.csv".format(path_suffix))
    prediction_file = os.path.join(pred_dir, "wepick_pred_{}.csv".format(path_suffix))

    if os.path.exists(model_prefix) == False: os.makedirs(model_prefix)
    if os.path.exists(log_dir) == False:  os.makedirs(log_dir)
    if os.path.exists(pred_dir) == False: os.makedirs(pred_dir)

    model = Model(user_count, item_count, cate_count, cate_list,
                  arch_dict,
                  FLAGS.use_item_embedding
                  )

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    if FLAGS.inspect_item_b:
      model.restore(sess, model_path)
      inspect_item_b(sess, model, item_b_pred_file)
      return 0

    if FLAGS.testonly:
      model.restore(sess, model_path)
      _predict(sess, model, prediction_file)
      return 0

    writer = tf.summary.FileWriter(log_dir, sess.graph)

    print('test_gauc: %.4f\t test_auc: %.4f' % _eval(sess, model, model_path))
    sys.stdout.flush()
    lr = 1.0
    start_time = time.time()
    for _ in range(FLAGS.num_epochs):

      random.shuffle(train_set)

      epoch_size = round(len(train_set) / FLAGS.batch_size)
      loss_sum = 0.0
      loss_count = 0
      for _, uij in DataInput(train_set, FLAGS.batch_size):
        summary, loss = model.train(sess, uij, lr)
        loss_sum += loss
        loss_count += 1

        if model.global_step.eval() % 100 == 0:
          test_gauc, Auc = _eval(sess, model, model_path)
          writer.add_summary(summary, model.global_step.eval())
          print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_GAUC: %.4f\tEval_AUC: %.4f' %
                (model.global_epoch_step.eval(), model.global_step.eval(),
                 loss_sum / loss_count, test_gauc, Auc))
          sys.stdout.flush()
          loss_sum = 0.0
          loss_count = 0

        if model.global_step.eval() % 336000 == 0:
          lr = 0.1

      print('Epoch %d DONE\tCost time: %.2f' %
            (model.global_epoch_step.eval(), time.time()-start_time))
      sys.stdout.flush()
      model.global_epoch_step_op.eval()

    writer.close()
    print('best test_gauc:', best_auc)
    sys.stdout.flush()

    return 0

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument(
      "--batch_size",
      type=int,
      default=1024,
      help="Batch size for training and eval.")
  parser.add_argument(
      "--num_epochs",
      type=int,
      default=10,
      help="number of training epochs.")
  parser.add_argument(
      "--predict_batch_size",
      type=int,
      default=512,
      help="Batch size for predicting.")
  parser.add_argument(
      "--predict_slot_after",
      type=int,
      default=21,
      help="When predicting, slots after this number will be considered as candidates.")
  parser.add_argument(
      "--learning_rate",
      type=float,
      default=0.001,
      help="Learning rate to be used during training.")
  parser.add_argument(
      "--testonly",
      action="store_true",
      default=False,
      help="Test Prediction Only. It will use the restored model.")
  parser.add_argument(
      "--inspect_item_b",
      action="store_true",
      default=False,
      help="Inspect item_b. and the result will be written to './item_b.csv'. (It eats up all options).")
  parser.add_argument(
      "--architecture",
      type=str,
      default="128_80_40",
      help="embed_dim,f1_dim,f2_dim")
  parser.add_argument(
      "--use_item_embedding",
      action="store_true",
      default=False,
      help="Model architecture including item embedding.")

  FLAGS, unparsed = parser.parse_known_args()

  sys.stdout.write('running options: {}\n'.format(str(FLAGS)))
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
