import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import deque
import math

fname = "../resource/gap_feature_3.txt"
with open(fname) as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [float(x.strip()) for x in content]
timeIndex = np.arange(0, len(content), 1)

windowLen = 7
windowStarts = [30, 60, 90, 130, 170]

fig, ax = plt.subplots()
ax.plot(timeIndex, content, linewidth=1)

max_limit = 0.1
max_update_alpha = 0.5
std_limit = 0.02
speaking = False

rectangles = []
# for start in windowStarts:
#     end = start + windowLen
#     window = np.array(content[start:end])
#     maxx = np.max(window)
#     std = np.std(window)
#     print("[{}, {}]\tmax: {} std: {}".format(start, end-1, maxx, std))
#     rectangles.append(
#         patches.Rectangle((start, 0), windowLen, max(content), linewidth=1, fill=False, color='red'))

gapDeque = deque([], maxlen=windowLen)
n = 0
for i in range(0, len(content)):
    gapFeature = content[i]
    gapDeque.append(gapFeature)
    if len(gapDeque) == gapDeque.maxlen:
        maxGap = max(gapDeque)
        meanGap = sum(gapDeque)/gapDeque.maxlen
        stdGap = math.sqrt(sum([(x-meanGap)**2 for x in gapDeque]) / gapDeque.maxlen)
        # the user is not speaking
        if not speaking:
            if gapFeature > max(max_limit * 1.5, 0.1):  # maybe is starting speaking, stop updating max_limit
                speaking = True
                start_speaking_index = i
            elif maxGap < max(max_limit, 0.1) and stdGap < std_limit:
                max_limit = max_limit * (1-max_update_alpha) + meanGap * 2 * max_update_alpha
                n += 1
                rectangles.append(patches.Rectangle((i-gapDeque.maxlen+1, 0), gapDeque.maxlen-1, 0.4, linewidth=0.1, fill=False, color='red'))
                plt.plot(i, max_limit, '.', color = 'green', markersize=1)
        else:  # the user is speaking
            if maxGap < min(max_limit, 0.1) and stdGap < std_limit*0.5:  # has stopped speaking
                end_speaking_index = i
                rectangles.append(
                    patches.Rectangle((start_speaking_index-gapDeque.maxlen+1, 0), end_speaking_index-start_speaking_index+gapDeque.maxlen-1, 0.4, linewidth=1,
                                      fill=False, color='black'))
                speaking = False
                max_limit = meanGap * 2

print(max_limit)
print(n)


for p in rectangles:
	ax.add_patch(p)

# ax.set_xlim([-20, 1100])
# ax.set_ylim([-20, 1940])

fig.set_size_inches(10, 3)
saveFigureName = '../resource/gap_feature.png'
fig.savefig(saveFigureName, dpi=300)
