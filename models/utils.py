import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import os
import collections
import sys
import subprocess
import pstats


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def profiling(script: str):
    subprocess.call(["/home/charles/anaconda3/envs/nlp/bin/python", "-m", "cProfile", "-o", "tempfile.txt", f'{script}'])
    p = pstats.Stats('tempfile.txt')
    return p.sort_stats('cumulative').print_stats(10)


def get_variable_sess(variable_name):
    return [v for v in tf.trainable_variables() if variable_name in v.name][0]


def special_bool(boolean: str):
    if type(boolean) == bool:
        return boolean
    else:
        if boolean == "True":
            return True
        else:
            return False


def profiling_tensorflow(sess, train_op, loss, ph, data):
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    _, loss_value = sess.run([train_op, loss], feed_dict={x_glove: x_glove_batch,
                                                          x_elmo_1: x_elmo_emb_batch_tr,
                                                          x_elmo_2: x_elmo_emb_batch_tr,
                                                          y: y_batch, prob: args.keep_prob},
                             options=options, run_metadata=run_metadata)

    # Create the Timeline object, and write it to a json file
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open(f'timeline_02_step_{j}.json', 'w') as f:
        f.write(chrome_trace)


def early_stopping(val_acc_list, patience=1):
    if len(val_acc_list) > 2:
        val_acc_l = val_acc_list[-(patience + 1):]
        print(val_acc_l)
        current_acc = val_acc_l[-1]
        l = [val_acc_l[i] for i in range(len(val_acc_l) - 1)]
        max_acc, amax_acc = max(l), (np.argmax(l) + 1)
        if current_acc < max_acc and (len(val_acc_l) - amax_acc) >= patience:
            return True
        else:
            return False
    else:
        pass


def summary_model(trainable_vars):
    # variables_names = [v.name for v in trainable_vars]
    # values = sess.run(variables_names)
    print("\n")
    print(120 * '*')
    print(120 * '*')
    print("MODEL SUMMARY".rjust(60))
    print(120 * '*' )
    print(120 * '*' + "\n")
    # for k, v in zip(variables_names, values):
    #     print("Variable: ", k)
    #     print("Shape: ", v.shape)
    #     print(v)
    total_parameters = 0
    for variable in trainable_vars:
        shape = variable.get_shape()

        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        print("Variable {} is of shape {} and has {} parameters".format(variable.name, shape,
                                                             variable_parameters) + "\n")
        total_parameters += variable_parameters
    print(120 * '*')
    print("The total number of parameters is: ", str(total_parameters))
    print(120 * '*' + "\n")


def classification_accuracy(logits, input_y):
    softmax_logits = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(softmax_logits, 1), tf.argmax(input_y, 1))
    # prediction = tf.argmax(softmax_logits, 1)
    # labels = tf.argmax(input_y, 1)
    # con_mtx = tf.confusion_matrix(labels, prediction, num_classes=num_class)
    # tp, fp, fn, tn = con_mtx[0][0], con_mtx[0][1], con_mtx[1][0], con_mtx[1][1]
    y_pred = tf.argmax(softmax_logits, 1)
    y_true = tf.argmax(input_y, 1)
    acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar("accuracy", acc)
    # F1 = (2 * tp) / (2 * tp + fp + fn)
    return acc


def f1_spec(y_true, y_pred):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average=None)


def classification_f1_score(logits, input_y):
    softmax_logits = tf.nn.softmax(logits)
    y_pred = tf.argmax(softmax_logits, 1)
    y_true = tf.argmax(input_y, 1)
    F1 = tf.py_func(f1_spec, [y_true, y_pred], Tout=tf.float64)
    return tf.reduce_mean([F1[0], F1[2]])


def metrics_f1_score(logits, input_y):
    y_pred = tf.argmax(logits, axis=-1, output_type=tf.int32)
    y_true = tf.argmax(input_y, 1)
    F1 = tf.contrib.metrics.f1_score(y_true, y_pred)
    return F1


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Epoch step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return True
    else:
        print(" No checkpoint found.")
        return False


def num_to_string_sst2(num):
    if num == 0:
        return "Negative"
    if num == 1:
        return "Positive"


def num_to_string_sst5(num):
    if num == 0:
        return "Very Negative"
    if num == 1:
        return "Negative"
    if num == 2:
        return "Neutral"
    if num == 3:
        return "Positive"
    if num == 4:
        return "Very Positive"


def write_logs_accuracy_sst(acc, args, logits, dic):
    softmax_logits = tf.nn.softmax(logits)
    input_y, x, x_org = dic["y_test"], dic["x_test_org"], dic["x_test_org_0"]
    predictions = tf.argmax(softmax_logits, 1, name="predictions")
    softmax_logits = softmax_logits.eval()
    labels = tf.argmax(input_y, 1)
    lens = []
    print("Analysis of Failed Cases")
    print("\n")
    count = 0
    success = 0
    for i, pred, labels in zip(range(len(x)), predictions.eval(), labels.eval()):
        if args.ds == "imdb":
            if pred == labels:
                success +=1
                if success % 1000:
                    print("***SUCCESS***")
                    print("Before Preprocesing: " + str(x_org[i]))
                    print("After Preprocessing " + str(x[i]), "Prediction is " + str(num_to_string_sst2(pred)),
                          "Truth is " + str(num_to_string_sst2(labels)) + " With Probs: " + str(softmax_logits[i]) + "\n")
        if pred != labels:
            count += 1
            print("Before Preprocesing: " + str(x_org[i]))

            if args.ds == "SST-2" or args.ds == "imdb":
                print("After Preprocessing " + str(x[i]), "Prediction is " + str(num_to_string_sst2(pred)),
                      "Truth is " + str(num_to_string_sst2(labels)) + " With Probs: " + str(softmax_logits[i]) + "\n")
                lens.append(len(x[i]))
            else:
                print("After Preprocessing " + str(x[i]), "Prediction is " + str(num_to_string_sst5(pred)),
                      "Truth is " + str(num_to_string_sst5(labels)) + " With Probs: " + str(softmax_logits[i]) + "\n")
                lens.append(len(x[i]))

    print(str(count) + " Errors, which is", str(round((count/len(x))*100, 2)) + '% of the test set' + "\n")
    print("Average sentence length in errors is: ",
          str(round(float(np.mean(lens)), 2)) + " words",
          " Average sentence length in test is: ",
          str(round(float(np.mean([len(el) for el in x])), 2)) + " words")
    print(40 * '*')
    print("Test accuracy is acc: ", acc)
    print(40 * '*')

    with open('time.txt', 'r') as f:
        time = f.readlines()[-1]
    time = round(int(time) / 60, 2)

    with open("results_new.txt", "a") as f:
        f.write("\n")
        f.write(60 * '*' + "\n")
        f.write("{}_model|batch_size_{}|"
                "lr_{}|nh_{}|keep_prob_{}|"
                "ds_{}|logdir_{}|max_length{}|"
                "input_only_{}|total_time_{}min|"
                "mean_{}|max_rank_{}|num_layers_{}".format(args.model, args.bs, args.lr, args.nh, args.keep_prob,
                                                           args.ds, args.logdir, args.ml, args.input_only,
                                                           str(time), args.mean, args.max_rank, args.num_layers) + "\n")
        f.write("General Test accuracy is: " + str(acc) + "\n")
        f.write(60 * '*')


def write_logs_accuracy(args, acc_train, acc_val, acc_test):
    with open('time.txt', 'r') as f:
        time = f.readlines()[-1]
    time = round(int(time) / 60, 2)

    with open("results_new.txt", "a") as f:
        f.write('\n')
        f.write("embs_{}|batch_size_{}|"
                "lr_{}|nh_{}|keep_prob_{}|"
                "ds_{}|logdir_{}|bptt_{}|"
                "total_time_{}min|"
                "num_layers_{}".format(args.model, args.bs, args.lr, args.n_hidden, args.keep_prob,
                                                           args.ds, args.logdir, args.bptt,
                                                           str(time), args.num_layers) + "\n")
        f.write("Train accuracy is: " + str(acc_train) + "\n")
        f.write("Val accuracy is: " + str(acc_val) + "\n")
        f.write("Test accuracy is: " + str(acc_test) + "\n")
        f.write(60 * '*')


def get_length_sent(dataset):
    lens = []
    for sent in dataset:
        lens.append(len(sent))
    return lens


def count_overlap(ds_name):
    test = np.load(f"/home/charles/Desktop/deep_nlp_research/data/datasets/{ds_name}/tmp/te_ids.npy")
    val = np.load(f"/home/charles/Desktop/deep_nlp_research/data/datasets/{ds_name}/tmp/val_ids.npy")
    train = np.load(f"/home/charles/Desktop/deep_nlp_research/data/datasets/{ds_name}/tmp/trn_ids.npy")
    train_unique = np.unique(train)
    test_unique = np.unique(test)
    val_unique = np.unique(val)
    nb_train_test = sum([a == el for el in tqdm(train_unique) for a in test_unique])
    print(f'There are {nb_train_test} in train and test' + '\n')
    nb_val_test = sum([a == el for el in val_unique for a in test_unique])
    print(f'There are {nb_val_test} in train and test' + '\n')
    nb_val_train = sum([a == el for el in val_unique for a in train_unique])
    print(f'There are {nb_val_train} in train and test' + '\n')
    return nb_train_test, nb_val_test, nb_val_train


def assert_disjoint(ds_name):
    test = np.load(f"/home/charles/Desktop/deep_nlp_research/data/datasets/{ds_name}/tmp/tok_te.npy")
    val = np.load(f"/home/charles/Desktop/deep_nlp_research/data/datasets/{ds_name}/tmp/tok_val.npy")
    train = np.load(f"/home/charles/Desktop/deep_nlp_research/data/datasets/{ds_name}/tmp/tok_trn.npy")
    train_unique = np.unique(train)
    test_unique = np.unique(test)
    val_unique = np.unique(val)
    els_tr_te, els_tr_val, els_val_te = [], [], []
    # for i, el1 in enumerate(train_unique):
    #     for el2 in test_unique:
    #         if len(el1) == len(el2):
    #             if el1 == el2:
    #                 els_tr_te.append(el1)
    # print(f'There are {len(els_tr_te)} in train and test' + '\n')
    # for i, el1 in enumerate(val_unique):
    #     for el2 in test_unique:
    #         if len(el1) == len(el2):
    #             if el1 == el2:
    #                 els_val_te.append(el1)
    # print(f'There are {len(els_val_te)} in val and test' + '\n')
    for i, el1 in enumerate(train_unique):
        for el2 in val_unique:
            if len(el1) == len(el2):
                if el1 == el2:
                    els_tr_val.append(el1)
                    print(i)
    print(f'There are {len(els_tr_val)} in train and val' + '\n')
    return els_tr_te, els_tr_val, els_val_te


def load_tok2id(ds_name):
    itos = np.load(f'data/datasets/{ds_name}/tmp/itos.pkl')
    return collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(itos)})


def padding_array(array, bptt):
    max_len = max([len(el) for el in array])
    num_cols = max(max_len, bptt)
    Xs = np.zeros((len(array), num_cols), dtype='int32')
    for i, sent_num in enumerate(array):
        num_zeros = num_cols - len(sent_num)
        n_sent_num = sent_num + num_zeros*[0]
        Xs[i] = n_sent_num
    Xs = Xs[:, :bptt]
    return Xs


def padding_array_elmo(array, bptt, emb_dim=1024):
    max_len = bptt
    Xs = np.ndarray((len(array), max_len, emb_dim), dtype='float32')
    for i, sent_num in enumerate(array):
        num_zeros = max_len - len(sent_num)
        n_sent_num = np.concatenate([sent_num, np.zeros((num_zeros, emb_dim))])
        Xs[i] = n_sent_num
    return Xs


def from_tokens_to_txt(tokens_batch, phase: str, id: int):
    current_sent = ' '
    with open(f'tmp_bert/bert_temp_{phase}_{id}.txt', 'w') as f:
        for sent in tokens_batch:
            new_sent = current_sent.join(sent)
            f.write(new_sent)
            current_sent = ' '
    with open(f'tmp_bert/bert_temp_{phase}_{id}.txt', 'r') as f:
        all_lines = f.readlines()
        with open(f'tmp_bert/bert_temp_{phase}_{id}.txt', 'w') as g:
            for line in all_lines[1:]:
                g.write(line[13:])


def load_train_val_test(ds_name):
    x_train, y_train = np.load(f'data/datasets/{ds_name}/tmp/trn_ids.npy'), \
                       np.load(f'data/datasets/{ds_name}/tmp/lbl_trn.npy')
    x_val, y_val = np.load(f'data/datasets/{ds_name}/tmp/val_ids.npy'), \
                   np.load(f'data/datasets/{ds_name}/tmp/lbl_val.npy')
    x_test, y_test = np.load(f'data/datasets/{ds_name}/tmp/te_ids.npy'), \
                   np.load(f'data/datasets/{ds_name}/tmp/lbl_te.npy')
    len_train, len_val, len_test = get_length(x_train), get_length(x_val), get_length(x_test)
    y_train, y_val, y_test = pd.get_dummies(np.squeeze(y_train)), pd.get_dummies(np.squeeze(y_val)), \
                             pd.get_dummies(np.squeeze(y_test))
    return x_train, y_train, x_val, y_val, x_test, y_test, len_train, len_val, len_test


def load_test_only(ds_name):
    x_test, y_test = np.load(f'data/datasets/{ds_name}/tmp/te_ids.npy'), \
                   np.load(f'data/datasets/{ds_name}/tmp/lbl_te.npy')
    len_test = get_length(x_test)
    y_test = pd.get_dummies(np.squeeze(y_test))
    return x_test, y_test, len_test


def get_length(array):
    return [len(el) for el in array]


def load_trn_toks(ds_name):
    return np.load(f'data/datasets/{ds_name}/tmp/tok_trn.npy')


def load_val_toks(ds_name):
    return np.load(f'data/datasets/{ds_name}/tmp/tok_val.npy')


def load_te_toks(ds_name):
    return np.load(f'data/datasets/{ds_name}/tmp/tok_te.npy')


def get_batch_size(len_voc):
    temp = [len_voc % i for i in range(1, 129)]
    ret = max([j for j, el in enumerate(temp) if el == 0])
    return ret + 1


def from_trees(ds, thr):
    import pytreebank
    data = pytreebank.import_tree_corpus(f"/home/charles/Desktop/deep_nlp_research/trees/{ds}.txt")
    dico = dict()
    final_sent, final_lab = [], []
    for el in data:
        for label, sentence in el.to_labeled_lines():
            dico[sentence] = label
    for el in dico.keys():
        if len(el.split()) > thr:
            final_sent.append(el)
            final_lab.append(dico[el])
    return pd.DataFrame({'lab': final_lab, 'sent': final_sent})


def ds_minus_one(ds_name):
    train = np.load(f'/home/charles/Desktop/deep_nlp_research/data/datasets/{ds_name}/tmp/lbl_trn.npy')
    train = train - 1
    np.save(f'/home/charles/Desktop/deep_nlp_research/data/datasets/{ds_name}/tmp/lbl_trn.npy', train)
    new_train = np.load(f'/home/charles/Desktop/deep_nlp_research/data/datasets/{ds_name}/tmp/lbl_trn.npy')

    val = np.load(f'/home/charles/Desktop/deep_nlp_research/data/datasets/{ds_name}/tmp/lbl_val.npy')
    val = val - 1
    np.save(f'/home/charles/Desktop/deep_nlp_research/data/datasets/{ds_name}/tmp/lbl_val.npy', val)
    new_val = np.load(f'/home/charles/Desktop/deep_nlp_research/data/datasets/{ds_name}/tmp/lbl_val.npy')

    test = np.load(f'/home/charles/Desktop/deep_nlp_research/data/datasets/{ds_name}/tmp/lbl_te.npy')
    test = test - 1
    np.save(f'/home/charles/Desktop/deep_nlp_research/data/datasets/{ds_name}/tmp/lbl_te.npy', test)
    new_test = np.load(f'/home/charles/Desktop/deep_nlp_research/data/datasets/{ds_name}/tmp/lbl_te.npy')
    print('New Train')
    print(new_train)
    print('New Val')
    print(new_val)
    print('New Test')
    print(new_test)


from random import choices


def bagging(x, y, prop_unique=False):
    x_y = np.concatenate([x, y], axis=1)
    bag = choices(x_y, k=len(x))
    x = np.array([np.array(el[:-1]) for el in bag])
    y = [el[-1] for el in bag]
    if prop_unique:
        joined_x = set()
        for el in x:
            new_num = ''
            num = [str(d) for d in el]
            joined_x.add(new_num.join(num))
        print("Propportion of unique is: ", len(joined_x) / len(x))
    return x, y


if __name__ == "__main__":
    assert_disjoint('semeval')

