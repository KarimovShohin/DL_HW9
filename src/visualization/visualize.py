import matplotlib.pyplot as plt

def plot_results(train_loss_hist, test_loss_hist):
    plt.plot(train_loss_hist, label='Train Loss')
    plt.plot(test_loss_hist, label='Test Loss')
    plt.legend()
    plt.show()