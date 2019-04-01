if __name__ == "__main__":
    import argparse
    import random
    import numpy as np
    import torch
    from models.bagging.mlp.mlp_train import Net
    from torchvision import datasets
    from collections import Counter
    import torchvision.transforms as transforms
    import torch.nn as nn
    from torch.utils.data.sampler import SubsetRandomSampler

    parser = argparse.ArgumentParser(description='TLBiLSTM network')
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU_id")
    parser.add_argument('--bagging', type=str, default=True, help="Bagging or Not")
    parser.add_argument('--ep', type=int, default=300, help="Number of Epochs")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--N', type=int, default=1, help="Number of Models")
    parser.add_argument('--T', type=float, default=1, help="Size of Dataset")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning Rate")
    parser.add_argument('--ds', type=str, default="iris", help="Dataset")
    parser.add_argument('--keep_prob', type=float, default=0.6, help="Dropout Rate")
    parser.add_argument('--num_layers', type=int, default=1, help="Number of BiLSTM layer")
    parser.add_argument('--max_checkpoints', type=int, default=1)
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()
    print("\n" + "Arguments are: " + "\n")
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    model = Net()

    model.load_state_dict(torch.load('model.pt'))

    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = 20
    # percentage of training set to use as validation
    valid_size = 0.2

    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()
    criterion = nn.CrossEntropyLoss()

    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    test_data = datasets.MNIST(root='data', train=False,
                               download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              num_workers=num_workers)

    model.eval()  # prep model for evaluation

    for data, target in test_loader:
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update test loss
        test_loss += loss.item() * data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)


    # # Testing
    # final_pred_ = []
    # for i in range(len(x_test)):
    #     c = Counter(predictions[:, i])
    #     final_pred_.append(c.most_common(1)[0][0])
    #
    # assert len(final_pred_) == len(y_test)
    #
    # acc = accuracy_score(y_test, final_pred_)
    #
    # with open(os.path.join('/home/charles/Desktop/deep_nlp_research/models/bagging/LogReg', 'results_bagging_logreg.txt'), 'a') as f:
    #     f.write(f'{args.bagging}|{args.N}|{args.T}|{str(acc)}' + '\n')