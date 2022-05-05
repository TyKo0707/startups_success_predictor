import pandas as pd
from environs import Env
import numpy as np
import matplotlib.pyplot as plt

env = Env()
env.read_env()
MAIN_DATASET_PATH = env.str("MAIN_DATASET_PATH")

df = pd.read_csv(MAIN_DATASET_PATH)

data = df.sort_values(by=['funding_total_usd'], ascending=False)

y = np.flipud(data.name.values[:20])
x = np.flipud(data.funding_total_usd.values[:20])
print((x[x.argmin()] - x[x.argmax()]) / 10)

plt.style.use('fast')
print(plt.style.available)

fig, ax = plt.subplots(figsize=(16, 9))

bar = ax.barh(y, x, color='#57a43a')
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)

ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)

ax.grid(b=True, color='gray',
        linestyle='-.', linewidth=0.5,
        alpha=0.2)


def gradientbars(bars):
    grad = np.atleast_2d(np.linspace(0, 1, 256))
    ax = bars[0].axes
    lim = ax.get_xlim() + ax.get_ylim()
    for bar in bars:
        bar.set_zorder(1)
        bar.set_facecolor("none")
        x, y = bar.get_xy()
        w, h = bar.get_width(), bar.get_height()
        ax.imshow(grad, extent=[x, x + w, y, y + h], aspect="auto", zorder=0)
    ax.axis(lim)


gradientbars(bar)

plt.ticklabel_format(style='plain', axis='x')
plt.ylabel('Company', labelpad=13)
plt.xlabel('Millions $', labelpad=13)
plt.title('Top 20 Funding Total raised by Company')
plt.show()
