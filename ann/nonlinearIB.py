import torch
import math
from progressbar import progressbar
import numpy as np
from torch.autograd import Variable

from ann.ibmodel import FCN, FCN2, FCN2_DB2
from ann.kde_estimation_mi import KDE_IXT_estimation
from ann.visualization import plot_results
from ann.visualization import init_results_visualization
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from dataloader.ninapro_loader import NinaPro_Dataset


class InformationBottleneck(torch.nn.Module):
    '''
    Implementation of the Kolchinsky et al. 2017 "Nonlinear Information Bottleneck"
    '''

    def __init__(self, input_shape, num_classes, network, K, beta, ib_type, logvar_t=-1.0, train_logvar_t=False):
        super(InformationBottleneck, self).__init__()

        self.input_shape = input_shape
        self.num_classes = num_classes

        self.HY = np.log(num_classes)  # in natts
        self.maxIXY = self.HY  # in natts
        self.varY = 0  # to be updated with the training dataset

        self.K = K
        self.beta = beta

        self.ib_type = ib_type

        self.train_logvar_t = train_logvar_t
        self.network = network(input_shape=input_shape, num_classes=num_classes, K=K,
                           logvar_t=logvar_t, train_logvar_t=self.train_logvar_t)
        self.network = self.network.to('cuda') if torch.cuda.is_available() else self.network

        self.ce = torch.nn.CrossEntropyLoss()

    def get_IXT(self, mean_t):
        '''
        Obtains the mutual information between the iput and the bottleneck variable.
        Parameters:
        - mean_t (Tensor) : deterministic transformation of the input
        '''

        IXT = KDE_IXT_estimation(self.network.logvar_t, mean_t)  # in natts
        self.IXT = IXT / np.log(2)  # in bits
        return self.IXT

    def get_ITY(self, logits_y, y):
        '''
        Obtains the mutual information between the bottleneck variable and the output.
        Parameters:
        - logits_y (Tensor) : deterministic transformation of the bottleneck variable
        - y (Tensor) : labels of the data
        '''
        HY_given_T = self.ce(logits_y, y.long())
        ITY = (self.HY - HY_given_T) / np.log(2)  # in bits
        return ITY, HY_given_T

    def get_loss(self, IXT_upper, ITY_lower, HY_given_T):
        '''
        Returns the loss function from the XXXX et al. 2019, "The Convex Information Bottleneck Lagrangian"
        Paramters: 
        - IXT (float) : Mutual information between X and T upper bound
        - ITY (float) : Mutual information between T and Y lower bound
        '''
        if self.ib_type == 'IB':
            loss = -1.0 * (ITY_lower - self.beta * IXT_upper)
        elif self.ib_type == 'nlIB':
            mi_penalty = IXT_upper ** 2
            loss = self.beta * mi_penalty - ITY_lower
        elif self.ib_type == '2OIB':
            mi_penalty = IXT_upper ** 2
            loss = self.beta * mi_penalty + HY_given_T ** 2
        else:
            loss = HY_given_T

        return loss

    def evaluate(self, logits_y, y):
        '''
        Evauluates the performance of the model (accuracy) for classification or (mse) for regression
        Parameters:
        - logits_y (Tensor) : deterministic transformation of the bottleneck variable
        - y (Tensor) : labels of the data
        '''

        with torch.no_grad():
            y_hat = y.eq(torch.max(logits_y, dim=1)[1])
            accuracy = torch.mean(y_hat.float())
            return accuracy

    def _evaluate_network(self, x, y, n_batches, IXT=0, ITY=0, loss=0, performance=0):

        logits_y = self.network(x)
        mean_t = self.network.encode_features(x, random=False)
        IXT_t = self.get_IXT(mean_t) / n_batches
        ITY_t, HY_given_T = self.get_ITY(logits_y, y)
        ITY_t /= n_batches
        HY_given_T /= n_batches

        loss_t = self.get_loss(IXT_t, ITY_t, HY_given_T) / n_batches

        IXT += IXT_t
        ITY += ITY_t
        loss += loss_t

        performance += self.evaluate(logits_y, y) / n_batches

        return IXT, ITY, loss, performance

    def evaluate_network(self, dataset_loader, n_batches):

        IXT = 0
        ITY = 0
        loss = 0
        performance = 0
        if n_batches > 1:
            for _, (x, y) in enumerate(dataset_loader):
                if torch.cuda.is_available():
                    x, y = x.cuda(), y.cuda()
                x, y = Variable(x), Variable(y)
                IXT, ITY, loss, performance = self._evaluate_network(x, y, n_batches, IXT, ITY, loss, performance)
        else:
            _, (x, y) = next(enumerate(dataset_loader))
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)
            IXT, ITY, loss, performance = self._evaluate_network(x, y, n_batches)

        return IXT.item(), ITY.item(), loss.item(), performance.item()

    def fit(self, train_set, test_set, n_epochs=200, optimizer_name='adam', learning_rate=0.0001, \
            learning_rate_drop=0.6, learning_rate_steps=10, train_batch_size=128, eval_rate=20, early_stopping_lim=15,
            verbose=True, visualization=True, logs_dir='.', figs_dir='.'):
        '''
        Trains the model with the training set with early stopping with the validation set. Evaluates on the test set.
        Parameters:
        - trainset (PyTorch Dataset) : Training dataset
        - validationset (PyTorch Dataset) : Validation dataset
        - testset (PyTorch Dataset) : Test dataset
        - n_epochs (int) : number of training epochs
        - optimizer_name (str) : name of the optimizer 
        - learning_rate (float) : initial learning rate
        - learning_rate_drop (float) : multicative learning decay factor
        - learning_rate_steps (int) : number of steps before decaying the learning rate
        - sgd_batch_size (int) : size of the SGD mini-batch
        - eval_rate (int) : the model is evaluated every eval_rate epochs
        - early_stopping_lim (int) : number of epochs for the validation set not to increase so that the training stops
        - verbose (bool) : if True, the evaluation is reported
        - visualization (bool) : if True, the evaluation is shown
        - logs_dir (str) : path for the storage of the evaluation
        - figs_dir (str) : path for the storage of the images of the evaluation
        '''

        # Definition of the training and test losses, accuracies and MI
        report = 0
        n_reports = math.ceil(n_epochs / eval_rate)
        train_loss_vec = np.zeros(n_reports)
        test_loss_vec = np.zeros(n_reports)
        train_performance_vec = np.zeros(n_reports)
        test_performance_vec = np.zeros(n_reports)
        train_IXT_vec = np.zeros(n_reports)
        train_ITY_vec = np.zeros(n_reports)
        test_IXT_vec = np.zeros(n_reports)
        test_ITY_vec = np.zeros(n_reports)
        epochs_vec = np.zeros(n_reports)
        early_stopping_count = 0
        loss_prev = math.inf
        # Data Loader
        n_sgd_batches = math.floor(len(train_set) / train_batch_size)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=train_batch_size,
            shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=train_batch_size, shuffle=False
        )
        # Prepare visualization
        if visualization:
            fig, ax = init_results_visualization(self.K)
            n_points = 10000
            pca = PCA(n_components=self.K//4)

        # Prepare name for figures and logs
        name_base = 'dimT-' + str(self.K) + '--beta-' + ('%.3f' % self.beta).replace('.', ',')

        # Definition of the optimizer
        if optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.network.parameters(), lr=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.network.parameters(), lr=learning_rate)
        elif optimizer_name == 'adadelta':
            optimizer = torch.optim.Adadelta(self.network.parameters(), lr=learning_rate)
        elif optimizer_name == 'adagrad':
            optimizer = torch.optim.Adagrad(self.network.parameters(), lr=learning_rate)
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        elif optimizer_name == 'asgd':
            optimizer = torch.optim.ASGD(self.network.parameters(), lr=learning_rate)
        learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, \
                                                                  step_size=learning_rate_steps,
                                                                  gamma=learning_rate_drop)

        best_test_accuracy = 0
        output_model = 'output/{}_{}_acc-{}.pkl'
        best_model = None
        # For all the epochs
        for epoch in range(n_epochs):

            print("Epoch #{}/{}".format(epoch, n_epochs))

            # Randomly sample a mini batch for the SGD
            for train_x, train_y in train_loader:
                if torch.cuda.is_available():
                    train_x, train_y = train_x.cuda(), train_y.cuda()
                train_x, train_y = Variable(train_x), Variable(train_y)
                # - Gradient descent
                optimizer.zero_grad()
                train_logits_y = self.network(train_x)
                train_ITY, train_HY_given_T = self.get_ITY(train_logits_y, train_y.long())
                train_mean_t = self.network.encode_features(train_x, random=False)
                train_IXT = self.get_IXT(train_mean_t)
                loss = self.get_loss(train_IXT, train_ITY, train_HY_given_T)
                loss.backward()
                optimizer.step()

            # Update learning rate
            learning_rate_scheduler.step()

            # Check for early stopping 
            with torch.no_grad():
                _, _, loss_curr, _ = self.evaluate_network(test_loader, len(test_set)//train_batch_size)
                if loss_curr >= loss_prev:
                    early_stopping_count += 1
                else:
                    early_stopping_count = 0
                if early_stopping_count >= early_stopping_lim:
                    break

            # If the results are improving
            if early_stopping_count == 0:
                # Report results
                if epoch % eval_rate == 0:
                    with torch.no_grad():
                        epochs_vec[report] = epoch

                        train_IXT_vec[report], train_ITY_vec[report], train_loss_vec[report], train_performance_vec[
                            report] = \
                            self.evaluate_network(train_loader, n_sgd_batches)
                        test_IXT_vec[report], test_ITY_vec[report], test_loss_vec[report], test_performance_vec[
                            report] = \
                            self.evaluate_network(test_loader, len(test_set)//train_batch_size)

                    if verbose:
                        print('\n\n** Results report **')
                        print(f'- Train | Test I(X;T) = {train_IXT_vec[report]} | {test_IXT_vec[report]}')
                        print(f'- Train | Test I(T;Y) = {train_ITY_vec[report]} | {test_ITY_vec[report]}')
                        print(
                            f'- Train | Test accuracy = {train_performance_vec[report]} | {test_performance_vec[report]}')
                    print(
                        f'- Current  | Best Test accuracy = {test_performance_vec[report]} | {best_test_accuracy}\n')
                    if best_test_accuracy < test_performance_vec[report]:
                        best_test_accuracy = test_performance_vec[report]
                        torch.save(self.network.state_dict(), output_model.format(self.ib_type, self.network.__class__, best_test_accuracy))
                        best_model = self.state_dict()
                    report += 1
                    # Visualize results and save results
                    if visualization:
                        with torch.no_grad():
                            _, (visualize_x, visualize_y) = next(enumerate(test_loader))
                            if torch.cuda.is_available():
                                visualize_x, visualize_y = visualize_x.cuda(), visualize_y.cuda()
                            visualize_x, visualize_y = Variable(visualize_x), Variable(visualize_y)

                            visualize_t = self.network.encode_features(visualize_x[:n_points], random=False)

                            plot_results(train_IXT_vec[:report], test_IXT_vec[:report],
                                         train_ITY_vec[:report], test_ITY_vec[:report],
                                         train_loss_vec[:report], test_loss_vec[:report],
                                         pca.fit_transform(visualize_t.cpu()), visualize_y[:n_points].cpu(), epochs_vec[:report],
                                         self.maxIXY, self.K,
                                         fig, ax)

                            plt.savefig(figs_dir + name_base + '--image.pdf', format='pdf')
                            plt.savefig(figs_dir + name_base + '--image.png', format='png')

                            print('The image is updated at ' + figs_dir + name_base + '--image.png')

                            np.save(logs_dir + name_base + '--hidden_variables', visualize_t.cpu())
                            np.save(logs_dir + name_base + '--labels', visualize_y.cpu())

                    # Save the other results
                    with torch.no_grad():
                        np.save(logs_dir + name_base + '--train_IXT', train_IXT_vec[:report])
                        np.save(logs_dir + name_base + '--validation_IXT', test_IXT_vec[:report])
                        np.save(logs_dir + name_base + '--train_ITY', train_ITY_vec[:report])
                        np.save(logs_dir + name_base + '--validation_ITY', test_ITY_vec[:report])
                        np.save(logs_dir + name_base + '--train_loss', train_loss_vec[:report])
                        np.save(logs_dir + name_base + '--validation_loss', test_loss_vec[:report])
                        np.save(logs_dir + name_base + '--train_performance', train_performance_vec[:report])
                        np.save(logs_dir + name_base + '--validation_performance', test_performance_vec[:report])
                        np.save(logs_dir + name_base + '--epochs', epochs_vec[:report])

