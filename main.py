import argparse
import os

import numpy as np
import h5py
import torch
from torch.optim import lr_scheduler
from ann.model import FCN as FCN
from ann.model import FCN2 as FCN2
from ann.ibmodel import FCN as ib_FCN
from ann.ibmodel import FCN2 as ib_FCN2
from ann.nonlinearIB import InformationBottleneck
from ann.train_and_test import train_ann, test_ann, test_ann_per_class
from snn.model import SpikingFCN
from dataloader.ninapro_loader import NinaPro_Dataset
from snn.spiking_neuron import IfNeuron, IFParameters
from snn.training_and_test import test_snn


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


parser = argparse.ArgumentParser(
    description='Run nonlinear IB (with Pytorch)',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--logs_dir', default='./results/logs/',
                    help='folder to output the logs')
parser.add_argument('--figs_dir', default='./results/figures/',
                    help='folder to output the images')
parser.add_argument('--models_dir', default='./results/models/',
                    help='folder to save the models')
parser.add_argument('--n_epochs', type=int, default=200,
                    help='number of training epochs')
parser.add_argument('--beta', type=float, default=0.005,
                    help='Lagrange multiplier (only for train_model)')
parser.add_argument('--beta_lim_min', type=float, default=0.0,
                    help='minimum value of beta for the study of the behavior')
parser.add_argument('--beta_lim_max', type=float, default=1.0,
                    help='maximum value of beta for the study of the behavior')
parser.add_argument('--n_betas', type=int, default=21,
                    help='Number of Lagrange multipliers (only for study behavior)')
parser.add_argument('--K', type=int, default=128 * 2,
                    help='Dimensionality of the bottleneck varaible')
parser.add_argument('--logvar_t', type=float, default=0.1,
                    help='initial log varaince of the bottleneck variable')
parser.add_argument('--sgd_batch_size', type=int, default=64,
                    help='mini-batch size for the SGD on the error')
parser.add_argument('--early_stopping_lim', type=int, default=20,
                    help='early stopping limit for non improvement')
parser.add_argument('--optimizer_name', choices=['sgd', 'rmsprop', 'adadelta', 'adagrad', 'adam', 'asgd'],
                    default='adam',
                    help='optimizer')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--learning_rate_drop', type=float, default=0.6,
                    help='learning rate decay rate (step LR every learning_rate_steps)')
parser.add_argument('--learning_rate_steps', type=int, default=10,
                    help='number of steps (epochs) before decaying the learning rate')
parser.add_argument('--train_logvar_t', action='store_true', default=False,
                    help='train the log(variance) of the bottleneck variable')
parser.add_argument('--eval_rate', type=int, default=1,
                    help='evaluate I(X;T), I(T;Y) and accuracies every eval_rate epochs')
parser.add_argument('--visualize', action='store_true', default=True,
                    help='visualize the results every eval_rate epochs')
parser.add_argument('--verbose', action='store_true', default=True,
                    help='report the results every eval_rate epochs')

simulation_config = {
    'random_seed': 0,
    'train_batch_size': 64,
    'train_epochs': 100,
    'to_save_the_best': True,
    'test_batch_size': 64,
    'use_gpu': torch.cuda.is_available(),
    'seq_length': 800,
    'max_firing_rate': 800,
    'dt': 0.001,
    'train_method': 'IB',  # IB, 2OIB, CE
}

if __name__ == '__main__':
    np.random.seed(simulation_config['random_seed'])
    torch.manual_seed(simulation_config['random_seed'])  # cpu
    torch.cuda.manual_seed(simulation_config['random_seed'])  # gpu

    args = parser.parse_args()
    np.set_printoptions(threshold=np.inf)

    file = h5py.File('dataset/NinaPro-DB1/DB1_S1_image.h5', 'r')
    imageData = file['imageData'][:]
    imageLabel = file['imageLabel'][:]
    file.close()

    # 随机打乱数据和标签
    N = imageData.shape[0]
    index = np.random.permutation(N)
    data = imageData[index, :, :]
    label = imageLabel[index]

    # 对数据升维,标签one-hot
    data = np.expand_dims(data, axis=1)
    # label = convert_to_one_hot(label, 52).T

    # 划分数据集
    N = data.shape[0]
    num_train = round(N * 0.8)
    print('num: ', N)
    X_train = data[0:num_train, :, :, :]
    Y_train = label[0:num_train]
    X_test = data[num_train:N, :, :, :]
    Y_test = label[num_train:N]

    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))

    train_set = NinaPro_Dataset(x=X_train, y=Y_train)
    test_set = NinaPro_Dataset(x=X_test, y=Y_test)

    dataset_name = 'NinaPro'
    logs_dir = os.path.join(args.logs_dir, dataset_name) + '/'
    figs_dir = os.path.join(args.figs_dir, dataset_name) + '/'
    models_dir = os.path.join(args.models_dir, dataset_name) + '/'
    os.makedirs(logs_dir) if not os.path.exists(logs_dir) else None
    os.makedirs(figs_dir) if not os.path.exists(figs_dir) else None
    os.makedirs(models_dir) if not os.path.exists(models_dir) else None

    train_loader = torch.utils.data.DataLoader(
        NinaPro_Dataset(x=X_train, y=Y_train),
        batch_size=simulation_config['train_batch_size'],
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        NinaPro_Dataset(x=X_test, y=Y_test),
        batch_size=simulation_config['test_batch_size'], shuffle=False
    )

    if simulation_config['train_method'] in ['IB', '2OIB']:
        # For ib_type options: IB, 2OIB
        ann = InformationBottleneck(input_shape=(1, 12, 10), num_classes=52, network=ib_FCN2,
                                    K=args.K, beta=args.beta, ib_type=simulation_config['train_method'],
                                    logvar_t=args.logvar_t, train_logvar_t=args.train_logvar_t)

        ann.fit(train_set, test_set, n_epochs=args.n_epochs, learning_rate=args.learning_rate,
                learning_rate_drop=args.learning_rate_drop, learning_rate_steps=args.learning_rate_steps,
                train_batch_size=args.sgd_batch_size,
                early_stopping_lim=args.early_stopping_lim, eval_rate=args.eval_rate,
                optimizer_name=args.optimizer_name, verbose=args.verbose,
                visualization=args.visualize, logs_dir=logs_dir, figs_dir=figs_dir)
    else:
        ann = ib_FCN2(input_shape=(1, 12, 10), num_classes=52)
        ann = ann.to('cuda') if simulation_config['use_gpu'] else ann
        #
        output_model = 'output/FCN2-CE_NinaPro-DB1_acc-{}.pkl'

        optimizer = torch.optim.Adam(ann.parameters(), lr=0.005)
        lr_sch = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.3)
        criterion = torch.nn.CrossEntropyLoss()

        # ann.load_state_dict(torch.load('output/IB_-FCN2-NinaPro-DB1_acc-0.7357046604156494.pkl'), strict=True)
        # accuracy = test_ann(model=ann, test_loader=test_loader, criterion=criterion, simulation_config=simulation_config)
        # accuracy = test_ann_per_class(model=ann, test_loader=test_loader, num_class=52)
        # print(accuracy)
        best_accuracy, best_epoch = 0, 1
        for i in range(simulation_config['train_epochs']):
            train_ann(model=ann, train_loader=train_loader, optimizer=optimizer, criterion=criterion, epoch=i + 1,
                      simulation_config=simulation_config)

            accuracy = test_ann(model=ann, test_loader=test_loader, criterion=criterion,
                                simulation_config=simulation_config)
            if simulation_config['to_save_the_best']:
                if best_accuracy < accuracy:
                    torch.save(ann.state_dict(), output_model.format(accuracy))
                    best_accuracy = accuracy
                    best_epoch = i + 1
            print('EPOCH: {}/{}, best test accuracy: {:.2f}% at epoch {}'.format(
                i + 1, simulation_config['train_epochs'], 100. * best_accuracy, best_epoch))
            lr_sch.step()
        # # if not to save the best model, we should save the model of last epoch.
        # if not simulation_config['to_save_the_best']:
        #     torch.save(ann.state_dict(), output_model)

        accuracy = test_ann_per_class(model=ann, test_loader=test_loader, num_class=52)
        print(accuracy)

    # Test SNN
    output_model = 'output/IB_-FCN2-NinaPro-DB1_acc-0.7357046604156494.pkl'
    device = torch.device('cuda')
    snn = SpikingFCN(input_shape=(1, 12, 10), num_classes=52,
                     spiking_neuron=IfNeuron, if_params=IFParameters(),
                     device=device,
                     seq_length=simulation_config['seq_length'],
                     max_firing_rate=simulation_config['max_firing_rate'],
                     dt=simulation_config['dt']).to(device)
    snn.load_state_dict(torch.load(output_model))
    test_snn(model=snn, test_loader=test_loader, simulation_config=simulation_config)
