import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def visual(x,y='fc',z='',annot=False):
    print(x.size())
    if annot:
        print(x)
    x=x.cpu()
    x=x.numpy()
    ax=sns.heatmap(x,xticklabels=[],yticklabels=[],annot=annot)
    ax.set_title(z,loc='center')
    plt.show()
    fig=ax.get_figure()
    fig.savefig(y+'.jpg')


def plot_loss(loss_values):
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'red', label='UrbanMC')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.grid(True,linestyle='--')
    plt.legend()
    plt.show()


with open('co2.txt', 'r') as file:
    lines = file.readlines()

loss_values1 = []
loss_values2 = []

for i,(line) in enumerate(lines):
    if i>100:
        break
    words = line.split()
    k=0
    for word in words:
        print(word)
        if k==0:
            loss_values1.append(float(word))
        else:
            loss_values2.append(float(word))
        k+=1


epochs = range(1, len(loss_values1) + 1)
plt.plot(epochs, loss_values1, 'red', label='ground truth')
plt.plot(epochs, loss_values2, 'blue', label='prediction')
plt.ylim(-1,50)
plt.title('(b) Flows of the same region at different timestamps')
plt.legend()
plt.show()



