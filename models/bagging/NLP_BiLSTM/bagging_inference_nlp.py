def RNN(dim_embs, args, trainable=True):

    if '2' in args.ds or 'imdb' in args.ds:
        num_class = 2
    else:
        num_class = 5
    if args.ds == 'semeval':
        num_class = 3

    # Defining Embeddings
    with tf.variable_scope("embedding"):
        W_in = tf.get_variable(name="W_in",
                               initializer=tf.constant(0.0,
                                                       shape=[dim_embs[0],
                                                              dim_embs[1]]),
                               trainable=trainable)
        embedding_init = W_in.assign(embedding_placeholder)
        embedded_chars = tf.nn.embedding_lookup(W_in, x)
        x_unstack = tf.unstack(embedded_chars, args.bptt, 1)

    with tf.variable_scope("biLSTM"):

        # Forward direction cell
        lstm_fw_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units=args.n_hidden / 2)
        lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, input_keep_prob=prob,
                                                     output_keep_prob=prob, seed=args.seed)
        # Backward direction cell
        lstm_bw_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units=args.n_hidden / 2)
        lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, input_keep_prob=prob,
                                                     output_keep_prob=prob, seed=args.seed)
        lstm_fw_cells = [lstm_fw_cell] * args.num_layers
        lstm_bw_cells = [lstm_bw_cell] * args.num_layers
        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(lstm_fw_cells, lstm_bw_cells, x_unstack,
                                                               dtype=tf.float32)

    with tf.variable_scope("fully_connected"):

        # Average all the hidden states across time
        outputs = tf.reduce_mean(outputs, axis=0)
        logits = tf.contrib.layers.fully_connected(outputs, num_class, activation_fn=None)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))
    tf.summary.scalar("loss", loss)

    return logits, loss, embedding_init, outputs


def optimizer_sched(args, sched=False):
    global_step = tf.Variable(0, trainable=False)
    if sched:
        lr_decayed = tf.train.cosine_decay_restarts(args.lr, global_step,
                                                    100)
        opt = tf.train.AdamOptimizer(lr_decayed, beta1=0.7, beta2=0.99)
    else:
        opt = tf.train.AdamOptimizer(args.lr, beta1=0.7, beta2=0.99)
    return opt, global_step


if __name__ == "__main__":
    import sys
    sys.path.append('/home/charles/Desktop/deep_nlp_research')
    from models.bilstm.embeddings import *
    from models.utils import *
    import numpy as np
    import argparse
    import random

    parser = argparse.ArgumentParser(description='TLBiLSTM network')
    parser.add_argument('--model', type=str, default="glove", help="Word Representation Model")
    parser.add_argument('--bs', type=int, default=256, help="Batch Size")
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU_id")
    parser.add_argument('--bagging', type=str, default=False, help="Bagging or Not")
    parser.add_argument('--bptt', type=int, default=100, help="BPTT")
    parser.add_argument('--debug', type=str, default=False, help="debug")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ep', type=int, default=300, help="Number of Epochs")
    parser.add_argument('--m', type=int, default=1, help="Size of Dataset")
    # parser.add_argument('--N', type=int, default=1, help="Number of Models")
    # parser.add_argument('--T', type=float, default=0.05, help="Size of Dataset")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning Rate")
    parser.add_argument('--ds', type=str, default="semeval", help="Dataset")
    parser.add_argument('--n_hidden', type=int, default=600, help="number of hidden states for BiLSTM")
    parser.add_argument('--keep_prob', type=float, default=0.6, help="Dropout Rate")
    parser.add_argument('--num_layers', type=int, default=1, help="Number of BiLSTM layer")
    parser.add_argument('--input_only', type=str, default=False, help="Add ELMo at the input and/or output of RNN")
    parser.add_argument('--max_checkpoints', type=int, default=1)
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()
    print("\n" + "Arguments are: " + "\n")
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"

    # Loading datasets and word embeddings
    os.chdir('/home/charles/Desktop/deep_nlp_research')
    (x_train, y_train, x_val, y_val, x_test,_, len_train, len_val, len_test), \
    tok2ids, trn_toks = load_train_val_test(args.ds), load_tok2id(args.ds), load_trn_toks(args.ds)
    x_test_padded = padding_array(x_test, args.bptt)
    y_test = np.load(f'data/datasets/{args.ds}/tmp/lbl_te.npy')
    y_pred = []
    for tr in range(args.m):
        tf.reset_default_graph()
        emb_mtx = load_embeddings(args.ds, args.model, tok2ids)
        assert emb_mtx.shape[0] == max(tok2ids.values()) + 1

        # Defining Placeholders
        num_class = y_val.shape[1]
        assert num_class == 2 or num_class == 5 or num_class == 3
        x = tf.placeholder(tf.int32, shape=[None, args.bptt])
        y = tf.placeholder(tf.float32, shape=[None, num_class])
        prob = tf.placeholder_with_default(1.0, shape=())
        lens = tf.placeholder(tf.int32, shape=[None])
        embedding_placeholder = tf.placeholder(tf.float32, shape=[emb_mtx.shape[0],
                                                                  emb_mtx.shape[1]])
        bs_ph = tf.placeholder(tf.int64)

        dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(args.bs).repeat()
        iter_test = dataset.make_initializable_iterator()
        n_batches_test = x_test.shape[0] // args.bs
        next_element_test = iter_test.get_next()
        logits, loss, embedding_init, outputs = RNN(emb_mtx.shape, args)
        saver = tf.train.Saver(max_to_keep=1)
        with tf.Session() as sess:
            logdir = f'tmp/bagging/bagging_{args.bagging}/m_{args.m}/{args.ds}/model_{tr}'
            load(saver, sess, logdir)
            # Defining Batches
            optimizer, global_step = optimizer_sched(args)[0], optimizer_sched(args)[1]
            if args.ds == 'sst2':
                predictions = sess.run(logits, feed_dict={x: x_test_padded})
            else:
                if args.ds == 'imdb' or args.ds == 'yelp2':
                    cl = 2
                elif args.ds == 'yelp5' or args.ds == 'sst5':
                    cl = 5
                elif args.ds == 'semeval':
                    cl = 3
                print('Number of Classes is: ', cl)
                predictions = np.ndarray((y_test.shape[0], cl))
                for j in range(1, n_batches_test + 1):
                    predictions[(j-1)*args.bs: j*args.bs] = sess.run(logits,
                                                                     feed_dict={x: x_test_padded[(j-1)*args.bs: j*args.bs]})

            predictions_ = np.argmax(predictions, axis=1)
            y_pred.append(predictions_)

    # Testing
    from sklearn.metrics import accuracy_score
    # from utils import f1_spec
    from collections import defaultdict
    final_pred = defaultdict(list)
    for i in range(len(y_pred[0])):
        for j in range(len(y_pred)):
            final_pred[i].append(y_pred[j][i])

    final_pred_ = []
    for key, value in final_pred.items():
        c = Counter(value)
        final_pred_.append(c.most_common(1)[0][0])
    rdm_idx = np.random.randint(0, len(y_test) - 1)
    print(final_pred[rdm_idx])
    print(final_pred_[rdm_idx])
    assert len(final_pred_) == len(y_test)

    if args.ds == 'semeval':
        F1 = f1_spec(y_test, final_pred_)
        print(len(y_test))
        F1_f = (F1[0] + F1[2]) / 2
        print('BAGGING F1 for semeval IS: ')
        print(F1_f)
        with open('/home/charles/Desktop/deep_nlp_research/models/bagging/NLP_BiLSTM/results_bagging.txt', 'a') as f:
            f.write(f'{args.bagging}|{args.m}|{str(F1_f)}' + '\n')

    else:
        print('BAGGING accuracy IS: ')
        print(len(y_test))
        acc = accuracy_score(y_test, final_pred_)
        print(acc)
        with open('/home/charles/Desktop/deep_nlp_research/models/bagging/NLP_BiLSTM/results_bagging.txt', 'a') as f:
            f.write(f'{args.bagging}|{args.m}|{str(acc)}' + '\n')