# -*- encoding: utf-8 -*-
import tensorflow as tf
from tensorflow.python import debug as tfdbg
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import time
import datetime
import json
import numpy as np
from collections import defaultdict
np.set_printoptions(formatter={'float': '{: .5f}'.format})
import scipy.sparse as spr
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

flags = tf.flags
FLAGS = flags.FLAGS

# model input
flags.DEFINE_string( 'dataset',                 "acm",
                     help="acm / imdb / mt", module_name="Input")
flags.DEFINE_boolean('onehot',                  True,
                     help="whether add one hot feature for those without features", module_name="Input")
# flags.DEFINE_boolean( 'small_mt',               False,
#                      help="whether remove p/s nodes for mt or not", module_name="Input")
flags.DEFINE_string( 'output',                  "./saved",
                     help="output dir path", module_name="Input")

# model structure
flags.DEFINE_string( 'embedding_model',         'Basemodel',
                     "Basemodel / Flatten / MultiHot", module_name="Archi")
flags.DEFINE_integer('layer',                   1,
                     help="num of layers", module_name="Archi")
flags.DEFINE_string( 'layer_aggr',              "sum",
                     help="", module_name="Archi")
flags.DEFINE_integer('embedding_size',          16,
                     help="embedding_size", module_name="Archi")
flags.DEFINE_integer('neighbor',                3,
                     help="neighbor sampling", module_name="Archi")
flags.DEFINE_string( 'attn',                    "simple",
                     help="variant: simple / multihead / selfattn / channel", module_name="Archi")
flags.DEFINE_boolean('simple_keepdim',          False,
                     help="whether use transformed input as output_mat in simple attention", module_name="Archi")
flags.DEFINE_boolean('cross',                   True,                           # 已废弃，请使用 "cluster_type"
                     help="use cross cluster or not", module_name="Archi")
flags.DEFINE_string( 'cluster_type',           'cross',
                     help="dual cluster or cross-cluster or single-cluster: dual/cross/single. ", module_name="Loss")
flags.DEFINE_string( 'norm',                    "bn",
                     help="bn: batch norm, or ln: layer norm", module_name="Archi")
flags.DEFINE_float(  'stu_v',                   1,
                     help="v in student-t of dualcluster", module_name="Archi")
flags.DEFINE_string( 'dist',                    "euclidean",
                     help="distance settings in dual cluster and belonging loss: euclidean / innerproduct", module_name="Archi")
flags.DEFINE_string( 'gradguide',               "softmax",
                     help="Way to calculate grad distribution in gradient guided loss: l1/stu/minmax/softmax ", module_name="Archi")

# loss coef:
flags.DEFINE_float(  'coef_cluster',            0,
                     help="coef of cluster loss. ", module_name="Loss")
flags.DEFINE_float(  'coef_dualcluster',        1,
                     help="coef of dual cluster loss. ", module_name="Loss")
flags.DEFINE_float(  'coef_reconst',            0,
                     help="0: not use reconstruction loss; >0: use", module_name="Loss")
flags.DEFINE_float(  'coef_belong',             1,
                     help="0: not use belonging loss (judge a node belong to a hyperedge); >0: use", module_name="Loss")
flags.DEFINE_float(  'coef_classify',           0,
                     help="0: not use classification loss; >0: use", module_name="Loss")
flags.DEFINE_float(  'coef_attnpair',           0,
                     help="0: not use attn pairwise hinge loss; >0: use", module_name="Loss")
flags.DEFINE_float(  'coef_grad',               10,
                     help="0: not use gradient guided attn loss; >0: use", module_name="Loss")
flags.DEFINE_float(  'coef_l2_emb',             0.1,
                     help="l2_coef for embedding matrix", module_name="Loss")
flags.DEFINE_float(  'coef_l2_net',             0,
                     help="l2_coef for network trainable parameters", module_name="Loss")
flags.DEFINE_integer('negnum',                  1,
                     help="negative sampling num for reconstruction loss", module_name="Loss")
flags.DEFINE_string( 'attnpair_norm',           "l2",
                     help="l1, l2", module_name="Loss")

# training parameter
flags.DEFINE_integer('epoch',                   200,
                     help="num of epochs", module_name="Opt")
flags.DEFINE_integer('n_init',                  5,
                     help="Number of time the model will be run with different centroid seeds. "
                          "The final results will be the best output of n_init consecutive runs in terms of inertia.", module_name="Opt")
flags.DEFINE_integer('pretrain_epoch',          0,
                     help="num of pretrain_epoch", module_name="Opt")
flags.DEFINE_float(  'dropout',                 0.5,
                     help="dropout", module_name="Opt")
flags.DEFINE_float(  'lr',                      5e-3,
                     help="learning rate", module_name="Opt")
flags.DEFINE_boolean('debug',                   False,
                     help="")
flags.DEFINE_boolean('save_emb',                False,
                     help="")
assert (FLAGS.coef_classify > 0 and FLAGS.coef_belong == 0) or (FLAGS.coef_classify == 0 and FLAGS.coef_belong > 0), \
    "one and only one of the [coef_classify, coef_belong] can be greater than 0. "

from Classifier import ClusterBaseline
from Utils import Logger, metrics


DEBUG = FLAGS.debug
dataset = FLAGS.dataset
# embedding_model = FLAGS.embedding_model
classifier_model = 'Cluster'

start_time_stamp = datetime.datetime.now().strftime('%m%d_%H%M')
dir_path = os.path.join(FLAGS.output, '{}/{}/{}'.format(
    dataset, FLAGS.embedding_model,
start_time_stamp
))

try:
    os.makedirs(dir_path)
except FileExistsError:
    postfix_num = 1
    while os.path.exists("{}_{}".format(dir_path, postfix_num)):
        postfix_num += 1
    dir_path = "{}_{}".format(dir_path, postfix_num)
    os.makedirs(dir_path)

checkpt_file = os.path.join(dir_path, 'checkpoint.ckpt')
log_path = os.path.join(os.path.dirname(checkpt_file), "print_logs.log")
sys.stdout = Logger(log_path)

from Utils.load_data import load_data
input_data = load_data(dataset, onehot_for_nofeature=FLAGS.onehot)

for para_groups in ['Input', 'Archi', 'Loss', 'Opt']:
    print('\n---------------- {} hyperparams -----------------'.format(para_groups))
    for flags_slice in FLAGS.flags_by_module_dict()[para_groups]:
        print("{:-<20s}{:->10s}  -----  HELP: {}"
              .format("{}  ".format(flags_slice.name), "  {}".format(flags_slice.value), flags_slice.help))
print()


# training procedure
# with tf.Graph().as_default():
runs_results = []
best_inertia = 9e9
for n_init_current in range(1, FLAGS.n_init+1):
    print('\n\n\n', "="*100, "\n\nRunning the {}/{}-th time. ".format(n_init_current, FLAGS.n_init))
    tf.reset_default_graph()

    print("embedding_model: {}\nclassifier_model: {}".format(FLAGS.embedding_model, classifier_model))
    print('Dataset: ' + dataset)
    clusterer = ClusterBaseline(FLAGS.embedding_model, input_data, FLAGS.embedding_size,)
    embedding_model = clusterer.model
    # paras = (np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    paras_shape = {v.name: v.get_shape().as_list() for v in tf.trainable_variables()}
    total_paras = np.sum([np.prod(v) for v in paras_shape.values()]).astype(np.int32)
    print("Total parameters : %d" % total_paras)
    print(json.dumps({k: str(v) for k, v in paras_shape.items()}, indent=4))
    embed_paras = int(embedding_model.feature_vocab_size * embedding_model.embedding_size)
    print("Embed parameters : %d" % embed_paras)
    print("Extra parameters : %d\n\n" % (total_paras - embed_paras))

    saver = tf.train.Saver()

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(os.path.join(dir_path, "tf_logs/"), sess.graph)
        sess.run(tf.global_variables_initializer())
        if DEBUG:
            sess = tfdbg.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tfdbg.has_inf_or_nan)

        if FLAGS.coef_reconst > 0 or FLAGS.coef_classify > 0 or FLAGS.coef_belong > 0:
            post_fix = ""
            for epoch in range(FLAGS.pretrain_epoch):
                # pretrain
                if FLAGS.coef_reconst > 0:
                    _, pretrain_loss, l2_loss = sess.run( [clusterer.pretrain_op_rec, clusterer.recon_loss, clusterer.lossL2_emb], )
                elif FLAGS.coef_classify > 0:
                    _, pretrain_loss, l2_loss = sess.run( [clusterer.pretrain_op_cls, clusterer.classification_loss, clusterer.lossL2_emb], )
                elif FLAGS.coef_belong > 0:
                    _, pretrain_loss, pretrain_loss_pos, pretrain_loss_neg, l2_loss = sess.run([
                        clusterer.pretrain_op_bel, clusterer.belong_loss,
                        clusterer.belong_pos_loss, clusterer.belong_neg_loss,
                        clusterer.lossL2_emb,
                    ])
                    post_fix = " | pos: {:.6f}, neg: {:.6f}".format(pretrain_loss_pos, pretrain_loss_neg)
                else:
                    pretrain_loss = "NONE"

                print('Run: {} | Epoch: {} | {:s} | PreTraining: loss = {:.9f}  L2_loss = {:.9f}'.format(
                    n_init_current, epoch,
                    time.strftime('%m-%d %H:%M:%S', time.localtime(time.time())), pretrain_loss, l2_loss),
                    post_fix,
                flush=True)


        attn_for_save = [0, None, None, None, None, None, None, None]
        best_results = defaultdict(lambda: -1e9)
        final_result = defaultdict(lambda: -1e9)
        for epoch in range(FLAGS.epoch):
            # train
            output_vars = [
                    # clusterer.monitored_tensors,
                    [],
                    clusterer.train_op, clusterer.loss, clusterer.losses,
                    clusterer.out_pred, clusterer.out_labe,
                    clusterer.inertia,
                    clusterer.merged_summary,
                ]
            if FLAGS.save_emb:
                output_vars.extend([
                    clusterer.model.ori_attn, clusterer.integrated_gradient,
                    clusterer.N2E_attns[clusterer.model.target_layer], clusterer.behavior_embedding,
                    clusterer.cluster_wgt_center, clusterer.cluster_emb_center, clusterer.whole_mask,
                ])

            ret = sess.run(output_vars)
            if FLAGS.save_emb:
                monitor_tensors, \
                _, train_loss, train_losses, preds, label, inertia, summary, \
                N2E_attn, N2E_grad, attn_emb, beha_emb, attn_center, baha_center, whole_mask = ret
            else:
                monitor_tensors, \
                _, train_loss, train_losses, preds, label, inertia, summary = ret

            train_acc, train_f1_macro = metrics.cluster_acc(y_true=label, y_pred=preds)
            train_nmi = metrics.cluster_nmi(y_true=label, y_pred=preds)
            train_ari = metrics.cluster_ari(y_true=label, y_pred=preds)
            if train_nmi > best_results['train_nmi']:
                best_results['train_acc'] = train_acc
                best_results['train_f1_macro'] = train_f1_macro
                best_results['train_nmi'] = train_nmi
                best_results['train_ari'] = train_ari

            final_result['train_acc'] = train_acc
            final_result['train_f1_macro'] = train_f1_macro
            final_result['train_nmi'] = train_nmi
            final_result['train_ari'] = train_ari
            train_losses_str = "  ".join(["{}: {:.5f}".format(n, l) for n,l in train_losses.items()])
            monitors = [" | Monitor:", *[np.array(m) for m in monitor_tensors]] if monitor_tensors else [""]
            print('Run: {} | Epoch: {} | {:s} | Training: {} | acc: {:.6f}, nmi: {:.6f}, ari: {:.6f} | inertia: {:.6f}'.format(
                n_init_current, epoch, time.strftime('%m-%d %H:%M:%S', time.localtime(time.time())),
                train_losses_str, train_acc, train_nmi, train_ari, inertia
            ), *monitors, flush=True)

            if FLAGS.save_emb and train_nmi > attn_for_save[0]:
                attn_for_save[0] = train_nmi
                attn_for_save[1] = N2E_attn
                attn_for_save[2] = N2E_grad
                # attn_emb, beha_emb, attn_center, baha_center
                attn_for_save[3] = attn_emb
                attn_for_save[4] = beha_emb
                attn_for_save[5] = attn_center
                attn_for_save[6] = baha_center
                attn_for_save[7] = whole_mask

            writer.add_summary(summary, epoch)
            writer.flush()

    runs_results.append([inertia, best_results, final_result])
    if FLAGS.save_emb:
        np.savez("attn.npz", attn=attn_for_save[1], grad=attn_for_save[2], label=label,
                 attn_emb=attn_for_save[3], beha_emb=attn_for_save[4],
                 attn_center=attn_for_save[5], baha_center=attn_for_save[6],
                 whole_mask=attn_for_save[7])
    gwriter = tf.summary.FileWriter(os.path.join(dir_path, 'my_graphs/'))
    gwriter.add_graph(sess.graph)


# Save result status:
print("Run {}/{} is Over. Log is output into: \n{}".format(n_init_current, FLAGS.n_init, log_path))
print("\n\nBest result in terms of inertia: ")
inertia, best_results, final_result = sorted(runs_results, key=lambda x: x[0])[0]
print("inertia: {:.6f},\nbest_results: {}\nfinal_result: {}".format(inertia, json.dumps(best_results), json.dumps(final_result)))

return_result = {
    "start_time_stamp": start_time_stamp,
    "dataset": FLAGS.dataset,
    "embedding_model": FLAGS.embedding_model,
    "task": "classification" if FLAGS.coef_classify > 0 else "cluster",
    "best_result": best_results,
    "final_result": final_result,
    "inertia": float(inertia),
    "log_path": log_path,
    "dir_path": dir_path,
    "paras": {
        "layer": FLAGS.layer,
        "embedding_size": FLAGS.embedding_size,
        "neighbor": FLAGS.neighbor,
        "stu_v": FLAGS.stu_v,
        "coef_dualcluster": FLAGS.coef_dualcluster,
        "coef_belong": FLAGS.coef_belong,
        "coef_classify": FLAGS.coef_classify,
        "coef_grad": FLAGS.coef_grad,
        "coef_l2_emb": FLAGS.coef_l2_emb,
        "epoch": FLAGS.epoch,
        "pretrain_epoch": FLAGS.pretrain_epoch,
        "lr": FLAGS.lr,
    }
}

result_path = os.path.join(FLAGS.output, FLAGS.dataset, "log_result.txt")
print("Result is output into: \n{}".format(result_path), flush=True)
with open(result_path, "a+") as f:
    f.write(json.dumps(return_result)+'\n')