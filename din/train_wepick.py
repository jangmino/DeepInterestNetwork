import os
import time
import pickle
import random
import numpy as np
import tensorflow as tf
import sys
import argparse

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
def _eval(sess, model):
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
    model.save(sess, 'save_path/ckpt')
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

def _predict(sess, model):
  with open('wepick_data.pkl', 'rb') as f:
    wepick_data = pickle.load(f)
    #deal_i_map = build_i_map(wepick_data['deal_map'])

  total_model = 0
  total_time = 0
  start = time.time()

  with open(FLAGS.pred_out_path, 'w') as pred_f:
    for _, uij in DataInputTest(test_set, FLAGS.predict_batch_size):
      outputs = []
      inf_start = time.time()
      users, histories, lengths = uij[0], uij[3], uij[4]
      predicted = model.test(sess, users, histories, lengths)
      inf_end = time.time()
      total_model += inf_end - inf_start

      for k in range(len(users)):
        u = wepick_data['user_key'][users[k]]
        hist = list(map(lambda x: wepick_data['deal_key'][x], histories[k][0:lengths[k]]))
        sort_i = np.argsort(predicted[k,:])
        sort_i = np.fliplr([sort_i])[0]
        order = list(filter(lambda x: x[1] >= FLAGS.predict_slot_after,
          map(lambda x: (wepick_data['deal_key'][x], wepick_data['deal_slot'][wepick_data['deal_key'][x]], predicted[k, x]), sort_i))
        )
        outputs.append((u, hist, order))

      for u, hist, order in outputs:
        h = "-".join(map(lambda x: str(x), hist))
        s = ":".join(map(lambda x: "{}/{}/{:.2f}".format(x[0], x[1], x[2]), order[:30]))
        pred_f.write("{},{},{}\n".format(u, h, s))
    #

  total_time += (time.time() - start)
  sys.stderr.write("Elapsed total {}: model {}\n".format(total_time, total_model))

def main(_):
  gpu_options = tf.GPUOptions(allow_growth=True)
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    model = Model(user_count, item_count, cate_count, cate_list, use_dice=True)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    if FLAGS.testonly:
      model.restore(sess, 'save_path/ckpt')
      _predict(sess, model)
      return 0

    writer = tf.summary.FileWriter('log', sess.graph)

    print('test_gauc: %.4f\t test_auc: %.4f' % _eval(sess, model))
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

        if model.global_step.eval() % 1000 == 0:
          test_gauc, Auc = _eval(sess, model)
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

    print('best test_gauc:', best_auc)
    sys.stdout.flush()

    return 0

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument(
      "--batch_size",
      type=int,
      default=512,
      help="Batch size for training.")
  parser.add_argument(
      "--num_epochs",
      type=int,
      default=10,
      help="number of training epochs.")
  parser.add_argument(
      "--predict_batch_size",
      type=int,
      default=64,
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
      default=True,
      help="Test Prediction Only. It will use the restored model.")
  parser.add_argument(
      "--pred_out_path",
      type=str,
      default="./wepick_pred.csv",
      help="Directory to write precition")

  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
