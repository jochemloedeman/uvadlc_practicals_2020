import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from train import train


def plot_curves(x, values, stddev, label, title):
    plt.plot(x, values[0], label="Sequence length 10")
    plt.fill_between(x, values[0] + stddev[0], values[0] - stddev[0], alpha=0.2)
    plt.plot(x, values[1], label="Sequence length 20")
    plt.fill_between(x, values[1] + stddev[1], values[1] - stddev[1], alpha=0.2)
    plt.xlabel('Steps')
    plt.ylabel(label)
    plt.legend()
    plt.title(title)
    plt.savefig(fname='peep_lstm.jpg', format='jpg', bbox_inches='tight', dpi=300)
    # plt.show()


def experiment():
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--dataset', type=str, default='bipalindrome',
                        choices=['randomcomb', 'bss', 'bipalindrome'],
                        help='Dataset to be trained on.')
    # Model params
    parser.add_argument('--model_type', type=str, default='peepLSTM',
                        choices=['LSTM', 'biLSTM', 'GRU', 'peepLSTM'],
                        help='Model type: LSTM, biLSTM, GRU or peepLSTM')
    parser.add_argument('--input_length', type=int, default=10,
                        help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1,
                        help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=256,
                        help='Number of hidden units in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=3000,
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)

    # Misc params
    parser.add_argument('--device', type=str, default="cuda:0",
                        help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--gpu_mem_frac', type=float, default=0.5,
                        help='Fraction of GPU memory to allocate')
    parser.add_argument('--log_device_placement', type=bool, default=False,
                        help='Log device placement for debugging')
    parser.add_argument('--summary_path', type=str, default="./summaries/",
                        help='Output path for summaries')

    seq_lengths = [10, 20]
    mean_accuracies = []
    standard_deviations = []
    for i in range(len(seq_lengths)):
        accuracies = []
        for seed in [0, 1, 2]:
            config = parser.parse_args()
            config.input_length = seq_lengths[i]
            accuracy = train(config)
            accuracies.append(accuracy)
        mean_accuracy = np.zeros(len(accuracies[0]))
        for seed in range(3):
            mean_accuracy += accuracies[seed]
        mean_accuracies.append(mean_accuracy / 3)
        stddev = []
        for j in range(len(mean_accuracy)):
            std = np.std(np.array([accuracies[0][j], accuracies[1][j], accuracies[2][j]]))
            stddev.append(std)
        standard_deviations.append(stddev)

    x = np.arange(len(mean_accuracies[0])) * 60
    plot_curves(x, mean_accuracies, standard_deviations, "Accuracy", "Accuracy for sequence length 10 and 20")
    print('hi')


if __name__ == '__main__':
    experiment()
