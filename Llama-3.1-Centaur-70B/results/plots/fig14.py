import pandas as pd
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import scienceplots
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import numpy as np
import nilearn
import nibabel as nib
from nilearn import plotting, datasets, surface, image
from nilearn.plotting import plot_stat_map
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.colors as mcolors

gs = gridspec.GridSpec(1, 2, width_ratios=[0.33333, 0.66666])
plt.style.use(['nature'])
fig = plt.figure(figsize=(7.08661, 7.08661/3))

# subplot 1
ax = fig.add_subplot(gs[:, 0])
image = plt.imread('test.png')
cax = ax.imshow(image)
ax.axis('off')
ax.set_title('Pearson correlation', y=1.19)

# Define a colormap
cmap = cmap=sns.color_palette("rocket_r", as_cmap=True)

# Create a norm (normalization of values)
norm = mcolors.Normalize(vmin=0, vmax=0.8)

# Create a ScalarMappable with colormap and norm
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Required for the colorbar

cbar = plt.colorbar(sm, ax=ax, orientation="vertical")
ax.text(-0.13, 1.32, 'a', transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')

# subplot 2
ax = fig.add_subplot(gs[:, 1])

layer = 2

df_centaur_tst = pd.read_csv('../results/feher2023rethinking/schaefer_tst_centaur_alignment.csv')
df_llama_tst = pd.read_csv('../results/feher2023rethinking/schaefer_tst_llama_alignment.csv')
df_random_tst = pd.read_csv('../results/feher2023rethinking/schaefer_tst_random_alignment.csv')
df_cognitive_tst = pd.read_csv('../results/feher2023rethinking/schaefer_tst_cognitive_alignment.csv')

print(df_cognitive_tst)

columns_to_keep = [
    ['Left Accumbens', 'Right Accumbens'], 
    ["b'7Networks_LH_Limbic_OFC_1'", "b'7Networks_LH_Default_pCunPCC_1'", "b'7Networks_RH_Limbic_OFC_1'", "b'7Networks_RH_Default_PFCdPFCm_1'"],
    ["b'7Networks_LH_SomMot_1'", "b'7Networks_LH_SomMot_2'", "b'7Networks_LH_SomMot_3'", "b'7Networks_LH_SomMot_4'", "b'7Networks_LH_SomMot_5'", "b'7Networks_LH_SomMot_6'"],
    ["b'7Networks_LH_Vis_1'", "b'7Networks_LH_Vis_2'", "b'7Networks_LH_Vis_3'", "b'7Networks_LH_Vis_4'", "b'7Networks_LH_Vis_5'", "b'7Networks_LH_Vis_6'", "b'7Networks_LH_Vis_7'", "b'7Networks_LH_Vis_8'", "b'7Networks_LH_Vis_9'", "b'7Networks_RH_Vis_1'", "b'7Networks_RH_Vis_2'", "b'7Networks_RH_Vis_3'", "b'7Networks_RH_Vis_4'", "b'7Networks_RH_Vis_5'", "b'7Networks_RH_Vis_6'", "b'7Networks_RH_Vis_7'", "b'7Networks_RH_Vis_8'"],
]

df_centaur_tst = df_centaur_tst.iloc[layer]
df_llama_tst = df_llama_tst.iloc[layer]
df_random_tst = df_random_tst.iloc[layer]
df_cognitive_tst = df_cognitive_tst.iloc[0]
print(df_cognitive_tst)
for i, column in enumerate(columns_to_keep):
    ax.bar(np.array([-1/5, 0, 1/5, 0.4]) + i, [df_centaur_tst[column].values.mean(), df_llama_tst[column].values.mean(), df_cognitive_tst[column].values.mean(), df_random_tst[column].values.mean()], alpha=0.8, color=['#69005f', '#ff506e', '#cbc9e2', 'grey'], width=1/5)
ax.set_ylabel('Pearson correlation')
ax.set_xticks([0.1, 1.1, 2.1, 3.1], ['Accumbens', 'Medial PFC', 'Motor Cortex', 'Visual Cortex'])

custom_lines_r2 = [Line2D([0], [0], color='#69005f', alpha=0.8, marker="o", linestyle='None', markersize=5), 
                   Line2D([0], [0], color='#ff506e', alpha=0.8, marker="o", linestyle='None', markersize=5),
                   Line2D([0], [0], color='#cbc9e2', alpha=0.8, marker="o", linestyle='None', markersize=5), 
                   Line2D([0], [0], color='grey', alpha=0.8, marker="o", linestyle='None', markersize=5),]
ax.legend(custom_lines_r2, ['Centaur', 'Llama', 'Cognitive model', 'Control'], frameon=False, ncols=4, bbox_to_anchor=(0.5, 1.3), loc='upper center')
ax.text(-0.06, 1.24, 'b', transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')

sns.despine()

plt.tight_layout()
plt.savefig('figures/fig14.pdf', bbox_inches='tight')

plt.show()
