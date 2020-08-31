import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.autograd import Variable
from tqdm import tqdm


def test_snn(model, test_loader, simulation_config):
    model.eval()

    total_output_record = None
    target_output = None
    t_l =tqdm(test_loader)

    for i, (data, target) in enumerate(t_l):
        if i >= 200:
            continue

        if isinstance(model.input_shape, int):
            data = data.view(-1, data.shape[1] * data.shape[2] * data.shape[3])
        if simulation_config['use_gpu']:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        spike_out, output_record = model(data)
        if total_output_record is None:
            total_output_record = output_record
        else:
            total_output_record = torch.cat((total_output_record, output_record), dim=1)
        if target_output is None:
            target_output = target.data.cpu()
        else:
            target_output = torch.cat((target_output, target.data.cpu()), dim=0)
        correct = total_output_record[-1, :].eq(target_output).sum()

        MT, FA = get_MT_and_FA(total_output_record, target_output)
        t_l.set_description('ACCURACY: {}/{} = {:.2f}%， MT: {}ms, FA: {:.2f}%'.format(
            correct, len(target_output), 100.0 * correct/len(target_output), MT, 100.0 * FA))

    plot_accuracy(total_output_record, target_output)


def get_MT_and_FA(total_output_record, target_output):
    correct_record = total_output_record[:, total_output_record[-1, :].eq(target_output)]
    shot_target = target_output[total_output_record[-1, :].eq(target_output)]
    total_correct_number = correct_record.shape[1]
    FA = 1.0 * total_correct_number / len(target_output)
    MT = -1
    for i in range(correct_record.shape[0]):
        correct = correct_record[i, :].eq(shot_target).sum()
        accuracy = 1.0 * correct / correct_record.shape[1]
        if accuracy >= 0.99 * FA:
            MT = i
            break

    return MT, FA


# Please ensure that the SNN simulates enough time steps.
def get_mt_and_fa_(total_accuracy_record):
    fa = total_accuracy_record[-1, 0] / total_accuracy_record[-1, 1] * 1.0
    fa_101 = fa * 0.99
    mt = 1
    for i in range(total_accuracy_record.shape[0]):
        mt = i + 1
        if total_accuracy_record[i, 0] / total_accuracy_record[i, 1] * 1.0 >= fa_101:
            break
    return mt, fa


def plot_accuracy(total_output_record, target_output):
    plt.xlabel('time [ms]')
    plt.ylabel('accuracy [%]')
    accuracy_record = torch.zeros((total_output_record.shape[0], ))
    for i in range(total_output_record.shape[0]):
        correct = total_output_record[i, :].eq(target_output).sum()
        accuracy = 1.0 * correct / len(target_output)
        accuracy_record[i, 0] = accuracy
    x = np.arange(1, accuracy_record.shape[0] + 1)
    plt.plot(x, accuracy_record)
    plt.show()
