# import sys
# sys.path.append('/home/charles/Desktop/deep_nlp_research')
# import time
# from models.bilstm.embeddings import *
# import numpy as np
# import argparse
# import random



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
    import tensorflow as tf
    import argparse
    parser = argparse.ArgumentParser(description='TLBiLSTM network')
    parser.add_argument('--model', type=str, default="glove", help="Word Representation Model")
    parser.add_argument('--bs', type=int, default=256, help="Batch Size")
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU_id")
    parser.add_argument('--bagging', type=str, default=True, help="Bagging or Not")
    parser.add_argument('--bptt', type=int, default=100, help="BPTT")
    parser.add_argument('--debug', type=str, default=False, help="debug")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ep', type=int, default=300, help="Number of Epochs")
    parser.add_argument('--m', type=int, default=20, help="Size of Dataset")
    # parser.add_argument('--N', type=int, default=1, help="Number of Models")
    # parser.add_argument('--T', type=float, default=0.5, help="Size of Dataset")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning Rate")
    parser.add_argument('--ds', type=str, default="sst2", help="Dataset")
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
    print(os.getcwd())
    os.chdir('/home/charles/Desktop/deep_nlp_research')
    (x_train, _, x_val, y_val, x_test, y_test, len_train, len_val, len_test), \
    tok2ids, trn_toks = load_train_val_test(args.ds), load_tok2id(args.ds), load_trn_toks(args.ds)
    x_train_padded, x_val_padded = padding_array(x_train, args.bptt), padding_array(x_val, args.bptt)
    y_train = np.load(f'data/datasets/{args.ds}/tmp/lbl_trn.npy')
    from sklearn.utils import shuffle
    x_train_padded, y_train = shuffle(x_train_padded, y_train, random_state=0)
    # x_train_padded, y_train = x_train_padded[:int(args.T * len(x_train_padded))], y_train[:int(args.T * len(y_train))]

    for tr in range(args.m):
        prop = 1 / args.m
        size_subset = int(prop * len(x_train))
        x_train_sub, y_train_sub = x_train_padded[tr * size_subset:(tr + 1) * size_subset], \
                                   y_train[tr * size_subset:(tr + 1) * size_subset]
        if special_bool(args.bagging):
            print('Bagging Activated')
            x_train_padded_, y_train_ = bagging(x_train_sub, y_train_sub, prop_unique=True)
            y_train_ = pd.get_dummies(y_train_)
            assert len(x_train_padded_) == len(y_train_)
        else:
            y_train_ = pd.get_dummies(np.squeeze(y_train_sub))
            x_train_padded_ = x_train_sub
        print('Len of X_train_padded is:', len(x_train_padded_))

        tf.reset_default_graph()
        with tf.Session() as sess:
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
            iter_val = dataset.make_initializable_iterator()
            iter_test = dataset.make_initializable_iterator()
            n_batches_val = x_val.shape[0] // args.bs
            n_batches_test = x_test.shape[0] // args.bs
            next_element_val = iter_val.get_next()
            next_element_test = iter_test.get_next()

            # Printing selected arguments
            if special_bool(args.debug):
                print("Debug Mode Activated")
                args.ep = 1

            iter_train = dataset.make_initializable_iterator()
            n_batches_train = x_train_padded_.shape[0] // args.bs
            next_element_train = iter_train.get_next()

            # Defining Batches
            optimizer, _ = optimizer_sched(args)[0], optimizer_sched(args)[1]

            logits, loss, embedding_init, outputs = RNN(emb_mtx.shape, args)

            # Defining main ops
            train_op = optimizer.minimize(loss)
            classification_accuracy_op = classification_accuracy(logits, y)

            if args.ds == 'semeval':
                classification_accuracy_op = classification_f1_score(logits, y)
            tf.summary.scalar('accuracy', classification_accuracy_op)

            # Allowing growing memory usage
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            saver = tf.train.Saver(max_to_keep=1)
            sess.run(tf.global_variables_initializer())
            summary_model(tf.trainable_variables())
            sess.run(embedding_init, feed_dict={embedding_placeholder: emb_mtx})

            # Initialize iterator for train data
            sess.run(iter_train.initializer, feed_dict={x: x_train_padded_, y: y_train_,
                                                        bs_ph: args.bs})
            sess.run(iter_val.initializer, feed_dict={x: x_val_padded, y: y_val,
                                                      bs_ph: args.bs})
            print("Session initialized")
            global_step, best_acc_val = 0, 0
            val_acc = []
            i = 0
            logdir = f'tmp/bagging/bagging_{args.bagging}/m_{args.m}/{args.ds}/model_{tr}'

            # Set up logging for TensorBoard.
            writer_train = tf.summary.FileWriter(os.path.join(logdir,'train'))
            writer_train.add_graph(tf.get_default_graph())
            writer_val = tf.summary.FileWriter(os.path.join(logdir,'val'))
            writer_val.add_graph(tf.get_default_graph())
            summary = tf.summary.merge_all()
            duration_training = 0
            for i in range(1, args.ep + 1):
                duration_epoch = 0
                avg_loss = 0
                print("Training...")
                print(4 * '***')
                print("Epoch {}".format(i))
                print(4 * '***')
                for j in range(1, n_batches_train + 1):
                    start_time = time.time()
                    x_batch, y_batch = sess.run(next_element_train)
                    _, loss_value, summary_train = sess.run([train_op, loss, summary],
                                             feed_dict={x: x_batch, y: y_batch, prob: args.keep_prob})
                    # writer_train.add_summary(summary_train, global_step)
                    avg_loss += loss_value
                    duration_step = time.time() - start_time
                    duration_epoch += duration_step
                    acc_train = sess.run(classification_accuracy_op, feed_dict={x: x_batch, y: y_batch})
                    print('step {:d} - loss = {:.3f} ||'.format(j, loss_value) +
                          " Training accuracy is: " + str(acc_train) + ' || ({:.3f} sec/step)'.format(
                        duration_step))
                    global_step += 1
                duration_training += duration_epoch
                print("\n" + 'Epoch {:d} - Average_Loss = {:.3f}, ({:.3f} sec/epoch)'.format(i,
                                                                                             avg_loss / n_batches_train,
                                                                                             duration_epoch))
                # Validation
                acc_val = 0
                for j in range(1, n_batches_val + 1):
                    x_batch, y_batch = sess.run(next_element_val)
                    acc_val += sess.run(classification_accuracy_op, feed_dict={x: x_batch, y: y_batch})
                    summary_val = sess.run(summary, feed_dict={x: x_batch, y: y_batch})
                    writer_val.add_summary(summary_val, global_step)

                acc_val = acc_val / n_batches_val
                print("Validation accuracy is: " + str(acc_val) + "\n")
                val_acc.append(acc_val)

                if acc_val > best_acc_val:
                    save(saver, sess, logdir, i)
                    best_acc_val = acc_val

                if early_stopping(val_acc, patience=args.patience):
                    print("EarlyStopping Activated with patience = {}, "
                          "validation accuracy list is: ".format(args.patience) + "\n")
                    print(val_acc)
                    best = np.argmax(val_acc) + 1
                    print(40 * '***')
                    print("Best model is at epoch: ", str(best), "and at step: ",
                          str(best * n_batches_train) + "\n")
                    print("Best accuracy is: ", str(val_acc[int(str(best)) - 1]) + "\n")
                    print(40 * '***')
                    break