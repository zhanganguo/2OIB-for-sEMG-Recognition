from torch.autograd import Variable
import torch
from tqdm import tqdm


def train_ann(model, train_loader, optimizer, criterion, epoch, simulation_config):
    model.train()
    t_l = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(t_l):
        if isinstance(model.input_shape, int):
            data = data.view(-1, data.shape[1] * data.shape[2] * data.shape[3])
        if simulation_config['use_gpu']:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        out = model(data)

        loss = criterion(out, target.long())
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            t_l.set_description('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #    epoch, batch_idx * len(data), len(train_loader.dataset),
            #    100. * batch_idx / len(train_loader), loss.data[0]))


def test_ann(model, test_loader, criterion, simulation_config):
    model.eval()
    test_loss = 0
    correct = 0
    t_l = tqdm(test_loader)
    for data, target in t_l:
        if isinstance(model.input_shape, int):
            data = data.view(-1, data.shape[1] * data.shape[2] * data.shape[3])
        if simulation_config['use_gpu']:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        out = model(data)
        test_loss += criterion(out, target.long()).item()
        pred = out.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    accuracy = 1.0 * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * accuracy))

    return accuracy


def test_ann_per_class(model, test_loader, num_class):
    model.eval()

    correct = list(0. for i in range(num_class))
    total = list(0. for i in range(num_class))
    for i, (images, labels) in enumerate(test_loader):
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

        output = model(images)

        prediction = torch.argmax(output, 1)
        res = prediction == labels
        for label_idx in range(len(labels)):
            label_single = int(labels[label_idx])
            correct[label_single] += res[label_idx].item()
            total[label_single] += 1
    acc_str = 'Accuracy: %f\n' % (sum(correct) / sum(total))
    for acc_idx in range(len(correct)):
        try:
            acc = correct[acc_idx] / total[acc_idx]
        except:
            acc = 0
        finally:
            acc_str += '\tclassID:%d\tacc:%f\t\n' % (acc_idx + 1, acc)

    return acc_str

