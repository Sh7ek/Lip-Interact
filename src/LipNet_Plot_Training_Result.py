import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

filename = "../resource/training_cmd_output.txt"
with open(filename) as f:
    content = f.readlines()
    content = [x.strip() for x in content]

list_accuracy = []
list_val_accuracy = []
for line in content:
    if "val_acc" in line:
        lineArr = line.split("-")
        acc_pair = lineArr[3]
        val_acc_pair = lineArr[5]
        accuracy = float(acc_pair.split(":")[1])
        val_accuracy = float(val_acc_pair.split(":")[1])
        list_accuracy.append(accuracy)
        list_val_accuracy.append(val_accuracy)

list_epoch = range(0, len(list_accuracy))

fig, ax = plt.subplots()
ax.plot(list_epoch, list_accuracy, linewidth=1, label='training')
ax.plot(list_epoch, list_val_accuracy, linewidth=1, label='validation')
plt.legend(['training', 'validation'])

ax.set_xlim([0, len(list_accuracy)])
ax.set_ylim([0, 1.1])
ax.grid(True)

fig.set_size_inches(4, 3)
saveFigureName = '../resource/training_accuracy.png'
fig.savefig(saveFigureName, dpi=300)




