# Try to create colormap from Arabas 2015
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, hex2color
#from matplotlib 

def plot_examples(cms):
    """
    helper function to plot two colormaps
    """
    np.random.seed(19680801)
    data = np.random.randn(30, 30)

    fig, axs = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
    for [ax, cmap] in zip(axs, cms):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
#    plt.show()

def plot_linearmap(cdict):
    newcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)
    rgba = newcmp(np.linspace(0, 1, 256))
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    col = ['r', 'g', 'b']
    for xx in [0.25, 0.5, 0.75]:
        ax.axvline(xx, color='0.7', linestyle='--')
    for i in range(3):
        ax.plot(np.arange(256)/256, rgba[:, i], color=col[i])
    ax.set_xlabel('index')
    ax.set_ylabel('RGB')
#    plt.show()
    
#%%

viridis = cm.get_cmap('viridis', 12)
print(viridis)
print(viridis(0.56))

print('viridis.colors', len(viridis.colors))
print('viridis.colors', viridis.colors)
print('viridis(range(12))', viridis(range(12)))
print('viridis(np.linspace(0, 1, 12))', viridis(np.linspace(0, 1, 12)))

#%%

N = 256
vals = np.ones((N, 4))
vals[:, 0] = np.linspace(90/256, 1, N)
vals[:, 1] = np.linspace(39/256, 1, N)
vals[:, 2] = np.linspace(41/256, 1, N)
newcmp = ListedColormap(vals)
#plot_examples([viridis, newcmp])
print("vals", newcmp.colors)

#%%
# 0 '#FFFFFF', 1 '#993399', 2 '#00CCFF', 3 '#66CC00', 4 '#FFFF00', 5 '#FC8727', 6 '#FD0000'

# from arabas gnuplot code

hex_colors = ['#FFFFFF', '#993399', '#00CCFF', '#66CC00',
              '#FFFF00', '#FC8727', '#FD0000']

rgb_colors = [hex2color(c) + tuple([1.0]) for c in hex_colors]

print(rgb_colors)
#print(mpl.colors.hex2rgb(hex_colors[0]))

#print(mpl.colors.rgb2hex((1.0,1.0,1.0,1.0), True) )

# print listed colormap => nearest neighbor interpolation!!
cmap_libcloudpplin = ListedColormap(rgb_colors)

#print("cmap_libcloudpp.colors", cmap_libcloudpp.colors)
   
plot_examples((viridis, cmap_libcloudpplin))

#%%
# create linear segmented colormap via dict
# need to separate (0.0, 1.0) into even distances
no_colors = len(rgb_colors)
dx_lin = 1.0/(no_colors-1)

cdict_lcpp_colors = np.zeros( (3, no_colors, 3) )

for i in range(3):
    cdict_lcpp_colors[i,:,0] = np.linspace(0.0,1.0,no_colors)
    for j in range(no_colors):
        cdict_lcpp_colors[i,j,1] = rgb_colors[j][i]
        cdict_lcpp_colors[i,j,2] = rgb_colors[j][i]

print(cdict_lcpp_colors)

cdict_lcpp = {"red": cdict_lcpp_colors[0],
              "green": cdict_lcpp_colors[1],
              "blue": cdict_lcpp_colors[2]}

cmap_lcpp = LinearSegmentedColormap('testCmap', segmentdata=cdict_lcpp, N=256)

plot_linearmap(cdict_lcpp)

plot_examples((viridis, cmap_lcpp))
