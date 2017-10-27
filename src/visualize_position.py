import matplotlib.pyplot as plt
import matplotlib.patches as patches

screen_center_x = 540
screen_center_y = 960
screen_width = 1080
screen_height = 1920

rectangles = []
rectangles.append(
	patches.Rectangle((screen_center_x - screen_width / 2, screen_center_y - screen_height / 2), screen_width,
	                  screen_height, linewidth=1, fill=False, color='black'))

node_bounds = [[540, 16, 114, 32],
               [539, 39, 93, 78],
               [540, 225, 1056, 282],
               [198, 182, 300, 135],
               [93, 182, 91, 135],
               [152, 182, 15, 135],
               [257, 182, 182, 135],
               [557, 194, 395, 159],
               [396, 173, 72, 49],
               [442, 225, 165, 49],
               [821, 190, 120, 120],
               [942, 169, 99, 77],
               [962, 229, 139, 42],
               [213, 304, 331, 61],
               [144, 1353, 264, 282],
               [408, 1353, 264, 282],
               [672, 1353, 264, 282],
               [936, 1353, 264, 282],
               [540, 1543, 1080, 84],
               [540, 1690, 1080, 291],
               [144, 1690, 264, 291],
               [408, 1690, 264, 291],
               [672, 1690, 264, 291],
               [936, 1690, 264, 291],
               [540, 1690, 1080, 291]]

for i in range(0, len(node_bounds)):
	node_bound = node_bounds[i]
	node_center_x = node_bound[0]
	node_center_y = node_bound[1]
	node_width = node_bound[2]
	node_height = node_bound[3]
	rectangles.append(
		patches.Rectangle((node_center_x - node_width / 2, node_center_y - node_height / 2), node_width, node_height,
		                  linewidth=1, fill=False, color='red'))

fig, ax = plt.subplots()
for p in rectangles:
	ax.add_patch(p)

ax.set_aspect('equal')
ax.set_xlim([-20, 1100])
ax.set_ylim([-20, 1940])
ax.set_ylim(ax.get_ylim()[::-1])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.tight_layout()
fig.set_size_inches(4, 6)
saveFigureName = 'node_position.png'
fig.savefig(saveFigureName, dpi=300)
