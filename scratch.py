import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
import numpy as np

# plt.figure()
# plt.bar(x=range(4), height=[1717, 2672, 1265, 1346], width=0.4)
# plt.xticks([0,1,2,3], ['Light', 'Medium', 'Heavy', 'Extreme'])
# plt.xlabel("Snowfall type")
# plt.ylabel("Number of frames")
# plt.show()

x = range(4)
y1 = [18.71, 22.19, 21.12, 20.47]
y2 = [19.09, 22.83, 21.96, 22.33]

ax = plt.subplot(111)
ax.plot(x, y1, 'o-g', label='Curriculum learning by snowfall level')
ax.plot(x, y2, 'o-b', label='Refine on all train set')
for i,j in list(zip(x,y1))+list(zip(x,y2)):
    ax.annotate(str(j),xy=(i,j))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xticks([0,1,2,3])
ax.set_xticklabels(['Light', 'Medium', 'Heavy', 'Extreme'])
ax.set_xlabel("Snowfall type")
ax.set_ylabel("SILog")
ax.legend()
plt.show()