import matplotlib.pyplot as plt
from model import UrbanZ

def scatter_plot_with_names(parameters, loss_values, names, colors=None, shapes=None):

    if colors is None:
        colors = ['b'] * len(parameters)
    if shapes is None:
        shapes = ['o'] * len(parameters)


    for i in range(len(parameters)):
        plt.scatter(parameters[i], loss_values[i], color=colors[i], marker=shapes[i])


        plt.annotate(names[i], (parameters[i], loss_values[i]), textcoords="offset points", xytext=(0, 10), ha='center')


    plt.xlabel('Parameter Size(M)')
    plt.ylabel('RMSE Loss')
    plt.xlim(0,7)
    plt.ylim(3.9,4.1)

    plt.grid(True,linestyle='--')

    plt.savefig('fig5.png', dpi=300)

    # 显示图形
    plt.show()


parameters = [4.79, 5.79, 11.28, 4.23, 2.63, 1.81]
loss_values = [4.054, 4.079, 4.042, 4.004, 4.085, 3.924]
names = ['VDSR', 'SRResNet', 'UrbanFM', 'FODE', 'IMDN', 'UrbanMC(ours)']
colors = ['r', 'g', 'b', 'c', 'm', 'y']
shapes = ['s', 'o', 'D', '^', 'x', '+']


scatter_plot_with_names(parameters, loss_values, names, colors, shapes)


model=UrbanZ()

total_params = sum(p.numel() for p in model.parameters())

print(total_params/1024/1024)
