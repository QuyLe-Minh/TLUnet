import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
# print(torch.log(torch.tensor([1.00000000e+00, 2.06541758e+02, 4.22427505e+02, 4.18862343e+02,
#                            2.47212595e+03, 4.33402164e+03, 3.91968337e+01, 1.50792414e+02,
#                            6.96975320e+02, 7.46346381e+02, 1.89924740e+03, 8.06596604e+02,
#                            1.58894967e+04, 1.27544505e+04])))
# checkpoint = torch.load("C:/Users/Admin/Downloads/Synapse_ckpt/Synapse_ckpt/model_final_checkpoint.model", map_location='cpu')
checkpoint = torch.load("C:/Users/Admin/Downloads/model_latest (1).model", map_location='cpu')
all_tr_losses, all_val_losses, all_val_losses_tr_mode, all_val_eval_metrics = checkpoint['plot_stuff']
epoch = checkpoint['epoch']-1

def plot_progress():
    """
    Should probably by improved
    :return:
    """
    font = {'weight': 'normal',
            'size': 18}

    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(30, 24))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    x_values = list(range(epoch + 1))

    ax.plot(x_values, all_tr_losses, color='b', ls='-', label="loss_tr")

    ax.plot(x_values, all_val_losses, color='r', ls='-', label="loss_val, train=False")

    if len(all_val_losses_tr_mode) > 0:
        ax.plot(x_values, all_val_losses_tr_mode, color='g', ls='-', label="loss_val, train=True")
    if len(all_val_eval_metrics) == len(x_values):
        ax2.plot(x_values, all_val_eval_metrics, color='g', ls='--', label="evaluation metric")

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax2.set_ylabel("evaluation metric")
    ax.legend()
    ax2.legend(loc=9)
    plt.show()
    
# plot_progress()
# print(all_val_eval_metrics[100:120])
print(checkpoint['epoch'])
print(all_val_eval_metrics[-20:])