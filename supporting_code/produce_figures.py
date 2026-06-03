import pandas as pd
import numpy as np
import scipy

import sys
sys.path.insert(0,'./')
from curlyBrace import curlyBrace
import seaborn as sns
import helper_functions

import matplotlib as mpl
import matplotlib.pyplot as plt

def lighten_color(color, amount):
    r, g, b = color[:3]
    return tuple((1 - amount) * np.array([r, g, b]) + amount * np.array([1, 1, 1]))
    
def darken_color(color, amount):
    r, g, b = color[:3]
    return tuple((1 - amount) * np.array([r, g, b]))
    
def figure_1(models,scatter_df,cmap,cutoff,save_file=None):
    plt.rcParams['text.usetex'] = True

    fig , ax = plt.subplots(ncols=2,figsize=(12,4),sharex=True,sharey=True)
    
    norm = mpl.colors.Normalize(vmin = scatter_df['Density'].min(), vmax = scatter_df['Density'].max())
    subplot_labels = ['A','B']
    for i,model in enumerate(models):
        model_rows = scatter_df['Model'].str.contains(model,na=False)
        df = scatter_df[model_rows]
    
        x = df['Prediction']
        y = df['Ground Truth']
    
        # plot colored points
        ax[i].scatter(x,y,c=df['Density'],norm=norm,cmap=cmap,s=5,rasterized=True)
    
        # add cutoff line
        ax[i].plot(ax[i].get_xlim(),[cutoff,cutoff],ls='--',color='w')
    
        # add curly braces
        c_brace_x = np.max(x + 0.25)
        c_brace_y_low = np.min(y - 0.25)
        c_brace_y_high = np.max(y + 0.25)
        k_r = 0.1/(c_brace_y_high-cutoff)*(cutoff-c_brace_y_low)
        
        curlyBrace(fig,ax[i],[c_brace_x,cutoff],[c_brace_x,c_brace_y_low],bool_auto=False,color='k')
        curlyBrace(fig,ax[i],[c_brace_x,c_brace_y_high],[c_brace_x,cutoff],k_r=k_r,bool_auto=False,color='k')
    
        # compute on/off correlations
        corr = np.corrcoef(x,y)[0,1]
        
        on_genes = y>cutoff
    
        on_corr = np.corrcoef(x[on_genes],y[on_genes])[0,1]
        print(f'Correlation of "On" genes: {on_corr}')
        
        off_corr = np.corrcoef(x[~on_genes],y[~on_genes])[0,1]
        print(f'Correlation of "Off" genes: {off_corr}')
    
        # add text
        ax[i].text(5.5,(c_brace_y_high-cutoff)/2+cutoff,f"``On'' genes\n$R$ = {on_corr:.3f}",verticalalignment='center',multialignment='center')
        ax[i].text(5.5,(cutoff-c_brace_y_low)/2+c_brace_y_low,f"``Off'' genes\n$R$ = {off_corr:.3f}",verticalalignment='center',multialignment='center')
        ax[i].text(-3,4.5,f'Overall correlation\n$R$ = {corr:.3f}',multialignment='center')

        ax[i].text(-0.05, 1.05, subplot_labels[i], transform=ax[i].transAxes, ha='left', fontsize=20)
        
        ax[i].axis('scaled')
        ax[i].set_xlim(right=9)
        
        ax[i].set_title(model)
        ax[i].set_xlabel('Prediction')
        if i == 0:
            ax[i].set_ylabel('Ground Truth')
    
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm = norm, cmap=cmap), ax=ax.ravel().tolist(),fraction=0.046, aspect=18)
    # plt.tight_layout()
    if save_file is not None:
        fig.savefig(save_file,bbox_inches='tight')
        
def figure_2(models,scatter_df,df_ON,df_OFF,error,cmap,model_colors,save_file = None):
    plt.rcParams['text.usetex'] = True
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(7, 8),sharey='row')
    sub_fsize = 20
    
    norm = mpl.colors.Normalize(vmin = scatter_df['Density'].min(), vmax = scatter_df['Density'].max())

    # ----------------------------------------------------------------- #
    # Subfigure A: Scatter density plot with varying "On" gene thresholds
    #  and corresponding correlations
    # ----------------------------------------------------------------- #

    # --- Plot colored points ---
    x = scatter_df['Prediction']
    y = scatter_df['Ground Truth']
    axes[0,0].scatter(x,y,c=scatter_df['Density'],norm=norm,cmap=cmap,s=5,rasterized=True)
    axes[0,0].axis('scaled')
    axes[0,0].set_xlim((np.min(x) - 1,np.max(x)+2))

    # draw two cutoff lines
    cutoff_left = -0.5
    axes[0,0].plot(axes[0,0].get_xlim(),[cutoff_left,cutoff_left],ls='--',color='w')
    
    cutoff_right = 1.4
    axes[0,0].plot(axes[0,0].get_xlim(),[cutoff_right,cutoff_right],ls='--',color='w')

    # --- draw curly braces ---
    # get highest point in scatter to define top edge of curly brace
    c_brace_y = np.max(y + 0.25)
    
    # add left curly brace
    on_genes = y>cutoff_left
    on_corr_left = np.corrcoef(x[on_genes],y[on_genes])[0,1]
    
    c_brace_x_left = np.min(x - 0.25)
    k_r = 0.055

    curlyBrace(fig,axes[0,0],[c_brace_x_left,cutoff_left],[c_brace_x_left,c_brace_y],k_r = k_r,bool_auto=False,color='k')
    num_samples_left = df_ON[(df_ON['Model'] == 'Enformer') & ((df_ON['Cutoff']*100).astype(int) == cutoff_left*100)]['% of genes'].mean()

    axes[0,0].text(c_brace_x_left-1,(c_brace_y-cutoff_left)/2+cutoff_left,f"``On'' genes\nthreshold = {cutoff_left}\n$R$ = {on_corr_left:.3f}\n{num_samples_left:.1f}\\% of genes",verticalalignment='center',multialignment='center',horizontalalignment='right')
    
    # add right curly brace
    on_genes = y>cutoff_right
    on_corr_right = np.corrcoef(x[on_genes],y[on_genes])[0,1]
    
    c_brace_x_right = np.max(x + 0.25)
    k_r = k_r/(c_brace_y-cutoff_right)*(c_brace_y-cutoff_left)
    
    curlyBrace(fig,axes[0,0],[c_brace_x_right,c_brace_y],[c_brace_x_right,cutoff_right],k_r = k_r,bool_auto=False,color='k')
    num_samples_right = df_ON[(df_ON['Model'] == 'Enformer') & ((df_ON['Cutoff']*100).astype(int) == cutoff_right*100)]['% of genes'].mean()

      
    axes[0,0].text(c_brace_x_right+1,(c_brace_y-cutoff_right)/2+cutoff_right,f"``On'' genes\nthreshold = {cutoff_right}\n$R$ = {on_corr_right:.3f}\n{num_samples_right:.1f}\\% of genes",verticalalignment='center',multialignment='center')
    
    # --- adjust plot features (like subfig label and axis visibility) ---
    axes[0,0].text(-0.25, 1.05, 'A', transform=axes[0,0].transAxes, ha='left', fontsize=sub_fsize)
    axes[0,0].axis('off')
    axes[0,1].axis('off')

    # ----------------------------------------------------------------- #
    # Subfigures B-E: Correlation and % of genes v. expression threshold
    # ----------------------------------------------------------------- #
    # pulling rows corresponding to Basenji, Enformer, and the random
    # samples of Basenji and Enformer from the larger dataframe
    random_rows = df_ON['Model'].str.contains('random',na=False)
    df_ON_models = df_ON[df_ON['Model'].isin(models)]
    df_OFF_models = df_OFF[df_OFF['Model'].isin(models)]

    # --- Subfigure B ---
    sns.lineplot(data=df_ON_models,x = 'Cutoff', y = 'Correlation', hue = 'Model', palette=model_colors, errorbar=error,ax=axes[1,0])
    sns.lineplot(data=df_ON[random_rows],x = 'Cutoff', y = 'Correlation', hue = 'Model', palette=model_colors, errorbar=error,linestyle='--',ax=axes[1,0])
    axes[1,0].set_xlabel('Minimum expression threshold')
    axes[1,0].set_ylabel('Ground truth v.~predicted\nPearson correlation')
    axes[1,0].get_legend().remove()
    axes[1,0].set_title("``On'' genes")
    axes[1,0].text(-0.05, 1.05, 'B', transform=axes[1,0].transAxes, ha='left', fontsize=sub_fsize)
    
    # --- Subfigure D ---
    sns.lineplot(data = df_ON_models, x = 'Cutoff', y = '% of genes', hue = 'Model', palette=model_colors, errorbar=error, ax=axes[2,0])
    sns.lineplot(data = df_ON[random_rows], x = 'Cutoff', y = '% of genes', hue = 'Model', palette=model_colors, errorbar=error,linestyle='--', ax=axes[2,0])
    axes[2,0].set_xlabel('Minimum expression threshold')
    axes[2,0].set_ylabel('\\% of genes')
    axes[2,0].get_legend().remove()
    axes[2,0].sharex(axes[1,0])
    axes[2,0].text(-0.05, 1.05, 'D', transform=axes[2,0].transAxes, ha='left', fontsize=sub_fsize)
    
    # --- Subfigure C ---
    random_rows = df_OFF['Model'].str.contains('random',na=False)
    sns.lineplot(data=df_OFF_models,x = 'Cutoff', y = 'Correlation', hue = 'Model', palette=model_colors, errorbar=error,ax=axes[1,1])
    sns.lineplot(data=df_OFF[random_rows],x = 'Cutoff', y = 'Correlation', hue = 'Model', palette=model_colors, errorbar=error,linestyle='--',ax=axes[1,1])
    axes[1,1].set_title("``Off'' genes")
    axes[1,1].sharex(axes[1,0])
    axes[1,1].set_xlabel('Maximum expression threshold')
    axes[1,1].text(-0.05, 1.05, 'C', transform=axes[1,1].transAxes, ha='left', fontsize=sub_fsize)

    # Legend placed with respect to subfig C
    handles, labels = axes[1,1].get_legend_handles_labels()
    axes[1,1].get_legend().remove()
    axes[0,1].legend(handles,labels,loc='center')

    # --- Subfigure E ---
    sns.lineplot(data = df_OFF_models, x = 'Cutoff', y = '% of genes', hue = 'Model', palette=model_colors, errorbar=error, ax=axes[2,1])
    sns.lineplot(data = df_OFF[random_rows], x = 'Cutoff', y = '% of genes', hue = 'Model', palette=model_colors, errorbar=error,linestyle='--', ax=axes[2,1])
    axes[2,1].sharex(axes[1,0])
    axes[2,1].set_xlabel('Maximum expression threshold')
    axes[2,1].text(-0.05, 1.05, 'E', transform=axes[2,1].transAxes, ha='left', fontsize=sub_fsize)
    axes[2,1].get_legend().remove()

    plt.tight_layout()

    # ----------------------------------------------------------------- #
    # Draw left set of arrows for correspondance of thresholds in 
    # Subfig A to expression thresholds in Subfigs B and D
    # ----------------------------------------------------------------- #
    # constants that will be used for the right arrow
    y_offset_corr = 0.04
    y_offset_num = 8

    # --- Convert points in plot coordinates to figure coordinates ---
    # Subfig A y-value
    p1 = fig.transFigure.inverted().transform(axes[0,0].transData.transform((0,cutoff_left)))

    # y-value of correlation for this threshold in Subfig B
    ppoint = fig.transFigure.inverted().transform(axes[1,0].transData.transform((cutoff_left,on_corr_left+y_offset_corr)))
    
    # Midpiont to avoid Subfig B's x-label
    # will be used for right arrow as well
    plabel = fig.transFigure.inverted().transform(axes[1,0].transAxes.transform((0,-0.12)))

    # y-value of % of genes for this threshold in Subfig D
    p2 = fig.transFigure.inverted().transform(axes[2,0].transData.transform((cutoff_left,num_samples_left+y_offset_num)))
    
    line = mpl.lines.Line2D(
        (p2[0], p2[0]),
        (p1[1], ppoint[1]),
        color='black', linestyle='--', linewidth=2, transform=fig.transFigure
    )
    fig.add_artist(line)
    line = mpl.lines.Line2D(
        (p2[0], p2[0]),
        (ppoint[1]-0.015, plabel[1]),
        color='black', linestyle='--', linewidth=2, transform=fig.transFigure
    )
    fig.add_artist(line)
    line = mpl.lines.Line2D(
        (p2[0], p2[0]),
        (plabel[1]-0.03,p2[1]),
        color='black', linestyle='--', linewidth=2, transform=fig.transFigure
    )
    fig.add_artist(line)

    # add arrow heads to subplots B and D
    axes[1,0].plot(*[cutoff_left,on_corr_left+y_offset_corr],'v',color='black',markersize='10')
    axes[2,0].plot(*[cutoff_left,num_samples_left+y_offset_num],'v',color='black',markersize='10')

    
    # ----------------------------------------------------------------- #
    # Draw right  set of arrows for correspondance of thresholds in 
    # Subfig A to expression thresholds in Subfigs B and D
    # ----------------------------------------------------------------- #
    
    # --- Convert points in plot coordinates to figure coordinates ---
    # Subfig A y-value
    p1 = fig.transFigure.inverted().transform(axes[0,0].transData.transform((0,cutoff_right)))
    
    # y-value of correlation for this threshold in Subfig B
    ppoint = fig.transFigure.inverted().transform(axes[1,0].transData.transform((cutoff_right,on_corr_right+y_offset_corr)))
    
    # y-value of % of genes for this threshold in Subfig D
    p2 = fig.transFigure.inverted().transform(axes[2,0].transData.transform((cutoff_right,num_samples_right+y_offset_num)))
    
    line = mpl.lines.Line2D(
        (p2[0], p2[0]),
        (p1[1], ppoint[1]),
        color='black', linestyle='--', linewidth=2, transform=fig.transFigure
    )
    fig.add_artist(line)
    line = mpl.lines.Line2D(
        (p2[0], p2[0]),
        (ppoint[1]-0.015, plabel[1]),
        color='black', linestyle='--', linewidth=2, transform=fig.transFigure
    )
    fig.add_artist(line)
    line = mpl.lines.Line2D(
        (p2[0], p2[0]),
        (plabel[1]-0.03,p2[1]),
        color='black', linestyle='--', linewidth=2, transform=fig.transFigure
    )
    fig.add_artist(line)
    
    # add arrow heads to subplots B and D
    axes[1,0].plot(*[cutoff_right,on_corr_right+y_offset_corr],'v',color='black',markersize='10')
    axes[2,0].plot(*[cutoff_right,num_samples_right+y_offset_num],'v',color='black',markersize='10')

    if save_file is not None:
        fig.savefig(save_file,bbox_inches='tight')
        
def figure_3(models,scatter_df,GMM_dist,track_corr,df_ON,df_OFF,error,cmap,model_colors,save_file = None):
    plt.rcParams['text.usetex'] = False
    fig = plt.figure(figsize=(13,15))

    # --- set up subplot axes ---
    # set up overall grid organizing figure
    outer_gs = mpl.gridspec.GridSpec(
        2, 6,
        figure=fig,
        height_ratios=[1, 1.5],
        width_ratios=[1, 1, 0.3, 1, 1, 1],  # col index 2 is a gap for upper set of 2x5 figures
        hspace=0.25
    )

    # split top row into two rows.  This way allows different control of 
    # hspace between top two rows vs. lower 2x2 plots.
    row1_gs = []
    for c in [0, 1, 3, 4, 5]:  # skip col 2
        row1_gs.append(
            mpl.gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer_gs[0, c], hspace=0.4 # put a 2x1 set of plots into the 1x1 figure grid spot 
            )
        )

    # set up lower grid of 2x2 subplots
    row2_gs = mpl.gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_gs[1,:5], wspace=0.2, hspace=0.3)

    # --- store subplots in data structures for passing into functions ---

    # list holding scatter axes
    scatter_ax = [fig.add_subplot(row1_gs[0][0])]
    for i in range(1,5):
        ax = fig.add_subplot(row1_gs[i][0], sharex=scatter_ax[0], sharey=scatter_ax[0])
        scatter_ax.append(ax)

    # list holding GMM v. corr axes
    GMM_corr_ax = [fig.add_subplot(row1_gs[0][1])]
    for i in range(1,5):
        ax = fig.add_subplot(row1_gs[i][1], sharex=GMM_corr_ax[0], sharey=GMM_corr_ax[0])
        GMM_corr_ax.append(ax)

    # 2x2 list holding running corr axes
    running_corr_ax = []
    for r in range(2):
        ax_row = []
        for c in range(2):
            ax = fig.add_subplot(row2_gs[r, c])
            ax_row.append(ax)
        running_corr_ax.append(ax_row)

    # all subplots share x
    running_corr_ax[0][1].sharex(running_corr_ax[0][0])
    running_corr_ax[1][0].sharex(running_corr_ax[0][0])
    running_corr_ax[1][1].sharex(running_corr_ax[0][0])

    # right side shares y with left side
    running_corr_ax[0][1].sharey(running_corr_ax[0][0])
    running_corr_ax[1][1].sharey(running_corr_ax[1][0])

    # --- plotting ---
    print('Overall model performance (Pearson correlation):')
    fig3_1(scatter_ax,models,scatter_df,cmap)
    
    print('GMM to Pearson correlations:')
    fig3_2(GMM_corr_ax,models,GMM_dist,track_corr,model_colors)

    fig3_3(running_corr_ax,df_ON,df_OFF,error,model_colors)

    # --- add subfigure labels ---
    
    scatter_ax[0].text(-0.45, 1.15, 'A',
                   transform=scatter_ax[0].transAxes, ha='left', fontsize=24) 
    scatter_ax[2].text(-0.45, 1.15, 'B',
                   transform=scatter_ax[2].transAxes, ha='left', fontsize=24) 
    # subfig C defined from GMM axis to align with subfig A label
    GMM_corr_ax[0].text(-0.45, -0.5, 'C',
                   transform=GMM_corr_ax[0].transAxes, ha='left', fontsize=24) 

    if save_file is not None:
        fig.savefig(save_file,bbox_inches='tight')
        
def fig3_1(axes,models,scatter_df,cmap):
    norm = mpl.colors.Normalize(vmin = scatter_df['Density'].min(), vmax = scatter_df['Density'].max())

    for i, model in enumerate(models):
        model_rows = scatter_df['Model'].str.contains(model,na=False)
        df = scatter_df[model_rows]
        x = df['Prediction']
        y = df['Ground Truth']

        corr = helper_functions.fast_pearson(x,y)
        print(f'\t{model}: {corr:.3f}') 
        # plot colored points
        axes[i].scatter(x,y,c=df['Density'],norm=norm,cmap=cmap,s=3,rasterized=True)

        # set title
        if '-' in model:
            # long synth distribution name needs to be split into two lines
            axes[i].set_title('\n'.join(model.split(' - ')))
        else:
            axes[i].set_title(model)

        # set x label
        if i < 2:
            axes[i].set_xlabel('Prediction')
        else:
            axes[i].set_xlabel('Synthetic "prediction"')

        # set y label for first subplot and hide otherwise
        if i == 0:
            axes[i].set_ylabel('Ground truth expression')
        elif i == 2:
            axes[i].set_ylabel('Synthetic "ground truth"')
        else:
            axes[i].tick_params(labelleft=False)

    # let axes settle
    plt.draw()

    # --- manually change xlims and ylims to force axis equal with
    # adjustable = 'datalim', which we can't use with sharex and sharey
    # this will make the axis edges match the plots in row2 of the fig

    # get bbox and compute ratio of width to height
    bbox = axes[0].get_position()
    fig_width, fig_height = axes[0].get_figure().get_size_inches()
    cell_ratio = (bbox.width * fig_width) / (bbox.height * fig_height)

    # the plots, when axis equal is applied, is too skinny.
    # So we know the span of y is larger and we need to add to the x limits
    xmin,xmax = axes[0].get_xlim()
    ymin,ymax = axes[0].get_ylim()

    span = ymax - ymin
    xcenter = (xmax + xmin) / 2

    # we only need to change one axis because of sharex sharey
    axes[0].set_xlim(xcenter - span * cell_ratio / 2, xcenter + span * cell_ratio / 2)
    
    # set ticks
    axes[0].xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
    axes[0].yaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
    
def fig3_2(axes,models,GMM_dist,track_corr,model_colors):
    for i,model in enumerate(models):
        print(f'\t{model}:')
        d = np.array(GMM_dist[model])
        c = np.array(track_corr[model])
        df = pd.DataFrame()
        df['d'] = d
        df['c'] = c
    
        # Find highest/lowest 1% of distance and correlation
        df['d_id'] = scipy.stats.rankdata(d) - 1
        df['c_id'] = scipy.stats.rankdata(c) - 1
        low = int(len(d)*.01)
        high = len(d)-low
        d_out = (df['d_id'] < low) | (df['d_id'] >= high)
        c_out = (df['c_id'] < low) | (df['c_id'] >= high)
        outliers = d_out | c_out
    
        # compute correlations
        corr_all,p_all = scipy.stats.pearsonr(d,c)
        corr_no_out, p_no_out = scipy.stats.pearsonr(df[~outliers]['d'],df[~outliers]['c'])
        print(f'\t\tCorrelation = {corr_all:.3f}, p = {p_all:.2e}')
        print(f'\t\tCorrelation = {corr_no_out:.3f}, p = {p_no_out:.2e} (Excluding top/bottom 1% distance and corr)')
    
        # plot scatters with outliers as x's
        sns.scatterplot(data = df[~outliers], x='d', y='c', color=model_colors[model],ax=axes[i])
        sns.scatterplot(data = df[outliers], x='d', y='c', marker='x', color=model_colors[model],ax=axes[i])
    
        # plot best-fit line
        m,b = np.polyfit(d,c,1)
        axes[i].axline((d[0],m*d[0]+b),linewidth=2,slope=m,color=(0.55,0.55,0.55))
    
        # plot corr and significance
        if i < 4:
            axes[i].text(1.05,0.47,f'$R$ = {corr_all:.3f},\np = {p_all:.2e}',fontsize=10)
        else:
            axes[i].text(1.05,0.86,f'$R$ = {corr_all:.3f},\np = {p_all:.2e}',fontsize=10)
    
        axes[i].set_xlabel('Distance between\nGaussian modes', multialignment='center')
        if i == 0 or i == 2:
            axes[i].set_ylabel('Ground truth v. predicted\nPearson correlation')
        else:
            axes[i].set_ylabel('')
            axes[i].tick_params(labelleft=False)
            
def fig3_3(axes,df_ON,df_OFF,error,model_colors):
    sns.lineplot(data=df_ON,x = 'Cutoff', y = 'Correlation', hue = 'Model', palette=model_colors, errorbar=error, linewidth=2, ax=axes[0][0])
    axes[0][0].set_title('"On" genes')
    axes[0][0].set_xlabel('Minimum expression threshold')
    axes[0][0].set_ylabel('Ground truth v. predicted\nPearson correlation')
    axes[0][0].get_legend().remove()
    
    sns.lineplot(data = df_ON, x = 'Cutoff', y = '% of genes', hue = 'Model', palette=model_colors, errorbar=error, linewidth=2, ax=axes[1][0])
    axes[1][0].set_xlabel('Minimum expression threshold')
    axes[1][0].set_ylabel('% of genes')
    axes[1][0].get_legend().remove()
    
    sns.lineplot(data=df_OFF,x = 'Cutoff', y = 'Correlation', hue = 'Model', palette=model_colors, errorbar=error, linewidth=2, ax=axes[0][1])
    axes[0][1].set_ylim(axes[0][0].get_ylim())
    axes[0][1].set_xlabel('Maximum expression threshold')
    axes[0][1].set_ylabel('')
    axes[0][1].set_title('"Off" genes')
    axes[0][1].legend(title='Model', loc='upper left', bbox_to_anchor=(1, 1))
    
    
    sns.lineplot(data = df_OFF, x = 'Cutoff', y = '% of genes', hue = 'Model', palette=model_colors, errorbar=error, linewidth=2, ax=axes[1][1])
    axes[1][1].set_xlabel('Maximum expression threshold')
    axes[1][1].set_ylabel('')
    axes[1][1].set_ylim(axes[1][0].get_ylim())
    axes[1][1].get_legend().remove()

def figure_4(variance_df, model_colors, save_file=None):
    plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots()
    
    models = variance_df['Model'].unique()
    x_positions = {model: i for i, model in enumerate(models)}
    
    # defining colors, markers, locations
    marker_map = {'model': '*', 'ground truth': 's', 'model prediction': '^'}
    size_map   = {'model': 200, 'ground truth': 100, 'model prediction': 100}
    offset = {'model': -0.1, 'model prediction':0.1,'ground truth':0.1}
    
    for model,group in variance_df.groupby('Model'):
        group = group.set_index('Source')
        
        # get general x location of this group of points
        x = x_positions[model]

        # plot dotted line
        x_line = x+offset['model prediction']
        y_triangle = group.loc['model prediction','Correlation']
        y_square = group.loc['ground truth','Correlation']
        ax.plot([x_line,x_line],[y_triangle,y_square], linestyle='--',color=model_colors[model])

        # plot star, triangle, square
        for source in group.index:
            # star is solid, square and triangle are white-filled
            facecolors = model_colors[model]
            if source != 'model':
                facecolors = 'white'

            # plot points
            y = group.loc[source, 'Correlation']
            ax.scatter(x+offset[source],y,marker=marker_map[source],s=size_map[source],color = model_colors[model],facecolors = facecolors)
    
    # build legend
    legend_handles = [
        plt.Line2D([0], [0], marker='*', linestyle='none', markerfacecolor='gray', markeredgecolor='gray', markersize=10, label='Continuous prediction'),
        plt.Line2D([0], [0], marker='s', linestyle='none', markerfacecolor='white', markeredgecolor='gray', markersize=8,  label='Naive from ground truth'),
        plt.Line2D([0], [0], marker='^', linestyle='none', markerfacecolor='white', markeredgecolor='gray', markersize=8,  label='Naive from model prediction'),
    ]
    plt.legend(handles=legend_handles,fontsize=14,loc='upper left',bbox_to_anchor=(1,1))

    # label axes
    plt.xticks(ticks=list(x_positions.values()),labels=list(x_positions.keys()), rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('$R$',fontsize=16)
    plt.xlim((-0.5,4.5))
    
    if save_file is not None:
        fig.savefig(save_file,bbox_inches='tight')

def figure_5(pub_freq, gene_corr, model_colors, save_file=None):
    plt.rcParams['text.usetex'] = False
    fig, ax = plt.subplots()
    # initialize data storage
    df = pd.DataFrame()
    fit = dict()
    spearman = dict()

    # collect correlation and publication counts for each model into single dataframe
    for i,key in enumerate(['Basenji','Enformer']):
        c = np.array(gene_corr[key])
        temp_df = pd.DataFrame({'pub_count':pub_freq,
                                'correlation':c,
                                'Model':[key]*len(pub_freq)})
        
        # remove genes with 0 publications so that we can take a log of the publication count
        temp_df = temp_df[temp_df['pub_count']>0]

        # add this model's correlations to dataframe
        df = pd.concat([df,temp_df])

        # compute linear fit of publication to performance
        fit[key] = scipy.stats.linregress(np.log10(temp_df['pub_count']), temp_df['correlation'])

        # compute Spearman correlation of publication to performance
        spearman[key] = scipy.stats.spearmanr(np.log10(temp_df['pub_count']), temp_df['correlation'])
        
    print(f'{np.sum(pub_freq == 0)} genes excluded from analysis for having 0 publications')
    
    # Define the bin edges
    bins = [1, 10, 100, 1000, 10000, np.inf]
    
    # Define the labels for each bin# Define the labels for each bin
    labels = ['1-10', '10-100', '100-1K', '1K-10K', '10K+']
    for i,l in enumerate(labels):
        n = np.sum((pub_freq >= bins[i]) & (pub_freq < bins[i+1]) )
        labels[i] += f'\nN={n}'

    # bin the publication frequency
    df['Publication Frequency'] = pd.cut(df['pub_count'], bins=bins, labels=labels, right=False)

    # plot split violin plot
    sns.violinplot(data = df, x = 'Publication Frequency', y = 'correlation', hue="Model", split=True, inner="quart",palette=model_colors,saturation=1,ax=ax)
    
    # plot linear regression
    x_val = np.arange(0.5,5.5,1)
    y_loc = -0.75
    for model in ['Enformer','Basenji',]:
        m,b = fit[model].slope, fit[model].intercept
        y = m*x_val+b
        print(f'{model}: Pearson R = {fit[model].rvalue:.4f}, p = {fit[model].pvalue:.4f}')
        plt.plot([0,1,2,3,4],y,color=darken_color(model_colors[model],0.2),linewidth=2)
        plt.text(0,y_loc,f'Spearman $R$={spearman[model].statistic:.3f}, p={spearman[model].pvalue:.3f}',color=model_colors[model])
        y_loc += 0.1

    ax.set_ylabel('Ground truth v. predicted\nPearson correlation')
    
    if save_file is not None:
        fig.savefig(save_file,bbox_inches='tight')

def figure_6(df, model_colors, save_file=None):
    plt.rcParams['text.usetex'] = False
    fig, ax = plt.subplots(figsize=(6.4,3.2))
    sns.barplot(df,x='Correlation',y='Model',capsize=.4, ax=ax)
    plt.bar_label(ax.containers[0], fmt='%.3f', padding=5)
    plt.draw()

    labels = [tick.get_text() for tick in ax.get_yticklabels()]
    
    for bar,label in zip(ax.patches, labels):
        if 'Basenji' in label:
            bar.set_facecolor(model_colors['Basenji'])
        else:
            bar.set_facecolor(model_colors['Enformer'])
        
    ax.set_xlabel('Pearson correlation across all positions per track')
    ax.set_xlim(right=0.81)
    
    if save_file is not None:
        fig = ax.get_figure()
        fig.savefig(save_file,bbox_inches='tight')
        
def figure_S1(df, model_colors, save_file=None):
    plt.rcParams['text.usetex'] = False
    
    # convert df to long format
    df_long = pd.melt(
        df,
        id_vars=['Tissue','Cancer','Distance'],
        value_vars=['Basenji', 'Enformer'],
        var_name='Model',
        value_name='Correlation'
    )
    
    # separate out cancer lines
    no_cancer_long = df_long[df_long['Cancer']=='No'].copy()
    no_cancer_long['Model'] = no_cancer_long['Model'].replace({'Basenji':'Basenji - Not Cancer','Enformer':'Enformer - Not Cancer'})
            
    mean_corr = df_long[df_long['Model']=='Enformer'].groupby('Tissue')['Correlation'].mean()
    value_counts = df_long[df_long['Model']=='Enformer']['Tissue'].value_counts(sort=False).rename('All')
    value_counts_no_cancer = df_long[(df_long['Model']=='Enformer') & (df_long['Cancer']=='No')]['Tissue'].value_counts(sort=False).rename('No Cancer')
    temp_df = pd.concat([mean_corr,value_counts,value_counts_no_cancer],axis=1).sort_values(by='Correlation',ascending=False)
    temp_df['No Cancer'] = temp_df['No Cancer'].fillna(0).astype(int)
    
    tissues = []
    for index,row in temp_df.iterrows():
        tissues.append(f'{index}\n{int(row['All'])} / {int(row['No Cancer'])}')
    
    
    fig, (axL,axR) = plt.subplots(ncols=2,figsize=(10,22),sharey=True,gridspec_kw={'wspace':0.5})
    
    model_colors['Basenji - Not Cancer'] = lighten_color(model_colors['Basenji'], 0.6)
    model_colors['Enformer - Not Cancer'] = lighten_color(model_colors['Enformer'], 0.6)
    
    sns.barplot(data=pd.concat([df_long,no_cancer_long]),x='Correlation',y='Tissue',hue='Model',hue_order=['Enformer','Basenji','Enformer - Not Cancer','Basenji - Not Cancer'],order=temp_df.index.values,ax=axL,errorbar='se',palette=model_colors)
    axL.set_xlim(left = 0.68)
    axL.invert_xaxis()
    axL.yaxis.tick_right()
    axL.yaxis.set_label_position("right")
    axL.tick_params(axis="y", pad=53)
    axL.set_yticks(range(len(tissues)))
    axL.set_yticklabels(tissues)
    plt.setp(axL.get_yticklabels(), ha="center")
    axL.set_ylabel('Tissue\n$N_\\text{all}$ / $N_\\text{not cancer}$',rotation='horizontal',ha='center',y=-0.01,labelpad=-45)
    axL.get_legend().remove()
    
    sns.barplot(data=pd.concat([df_long,no_cancer_long]),x='Distance',y='Tissue',hue='Model',hue_order=['Enformer','Basenji','Enformer - Not Cancer','Basenji - Not Cancer'],order=temp_df.index.values,ax=axR,errorbar='se',palette=model_colors)
    axR.set_xlim(left = 1.25)
        
    sns.move_legend(axR,'upper left',bbox_to_anchor=(1,1))
    
    if save_file is not None:
        fig.savefig(save_file,bbox_inches='tight')

def figure_S2(tissue_corr, tissue_GMM, tissue_corr_2, tissue_GMM_2, model_colors, save_file=None):
    plt.rcParams['text.usetex'] = False

    fig,ax = plt.subplots(nrows=2, ncols = 4, figsize = (10,6), sharex=True,sharey=True)
    plt.subplots_adjust(hspace=0.4)
    
    for row,method in enumerate(['Average GMM distance','Collate ground truth then compute new GMM']):
        print(f'Method: {method}:')
        for i,model in enumerate(tissue_corr.keys()):
            print(f'\t{model}:')
            if row == 0:
                d = np.array(tissue_GMM[model])
                c = np.array(tissue_corr[model])
            else:
                d = np.array(tissue_GMM_2[model])
                c = np.array(tissue_corr_2[model])
                
            df = pd.DataFrame()
            df['d'] = d
            df['c'] = c
        
            # compute correlations
            corr_all,p_all = scipy.stats.pearsonr(d,c)
            print(f'\t\tCorrelation = {corr_all:.3f}, p = {p_all:.2e}')
        
            # plot scatters with outliers as x's
            sns.scatterplot(data = df, x='d', y='c', color=model_colors[model],ax=ax[row,i])
        
            # plot best-fit line
            m,b = np.polyfit(d,c,1)
            ax[row,i].axline((d[0],m*d[0]+b),linewidth=2,slope=m,color=(0.55,0.55,0.55))
        
            ax[row,i].set_xlabel('Distance between\nGaussian modes', multialignment='center')
            if i == 0:
                ax[row,i].set_ylabel('Ground truth v. predicted\nPearson correlation')
            else:
                ax[row,i].set_ylabel('')
                ax[row,i].tick_params(labelleft=False)

            if model[:7] == 'Basenji':
                ax[row,i].text(1.6,0.85,f'$R$ = {corr_all:.3f},\np = {p_all:.2e}',fontsize=10)
            else:
                ax[row,i].text(1.675,0.765,f'$R$ = {corr_all:.3f},\np = {p_all:.2e}',fontsize=10)
        
            ax[row,i].set_title(model)

    ax[0,0].text(-0.6, 1.15, 'A',
                   transform=ax[0,0].transAxes, ha='left', fontsize=24) 
    ax[1,0].text(-0.6, 1.15, 'B',
                   transform=ax[1,0].transAxes, ha='left', fontsize=24) 

    if save_file is not None:
        fig.savefig(save_file,bbox_inches='tight')

def figure_S3(ground_truth,predictions,gt_cutoff,pred_cutoff,cmap,save_file=None):
    plt.rcParams['text.usetex'] = True

    df = helper_functions.estimate_distribution_density(predictions,ground_truth,'')

    fig , ax = plt.subplots(nrows=2, ncols=2,figsize=(8,8),sharey=True)
    fig.subplots_adjust(hspace=0.3)
    
    norm = mpl.colors.Normalize(vmin = df['Density'].min(), vmax = df['Density'].max())
    subplot_labels = ['A','B']

    # ------------------------------------------------------------------ #
    # Upper Left: Enformer scatter with HORIZONTAL line at gt cutoff
    # ------------------------------------------------------------------ #  
    x = df['Prediction']
    y = df['Ground Truth']

    # plot colored points
    ax[0,0].scatter(x,y,c=df['Density'],norm=norm,cmap=cmap,s=5,rasterized=True)

    # add cutoff line
    ax[0,0].axhline(gt_cutoff, ls='--', color='w')

    ax[0,0].axis('scaled')

    ax[0,0].set_xlabel('Enformer prediction')
    ax[0,0].set_ylabel('Ground truth')
    ax[0,0].text(-0.05, 1.05, 'A',
               transform=ax[0,0].transAxes, ha='left', fontsize=20)    
    
    # ------------------------------------------------------------------ #
    # Upper Right: Violin — class by GT cutoff, Y = ground truth values
    # ------------------------------------------------------------------ #
    pred_class = (y >= gt_cutoff).astype(int)  # 0 = off, 1 = on

    vp_ur = sns.violinplot(x = pred_class, y = df['Ground Truth'],ax=ax[0,1],inner="quart",color='gray')
    ax[0,1].axhline(gt_cutoff, ls='--', color='gray', alpha=0.7)
    ax[0,1].set_xticks([0, 1])
    ax[0,1].set_xticklabels(['Off (GT $<$ cutoff)', 'On (GT $\\geq$ cutoff)'])
    ax[0,1].set_xlabel('Naive prediction')
    ax[0,1].set_ylabel('Ground Truth')
    # ax[0,1].set_title('GT-based classification')
    ax[0,1].text(-0.05, 1.05, 'B',
               transform=ax[0,1].transAxes, ha='left', fontsize=20)    
    
    # ------------------------------------------------------------------ #
    # Lower Left: Enformer scatter with VERTICAL line at prediction cutoff
    # ------------------------------------------------------------------ #  
    x = df['Prediction']
    y = df['Ground Truth']

    # plot colored points
    ax[1,0].scatter(x,y,c=df['Density'],norm=norm,cmap=cmap,s=5,rasterized=True)

    # add cutoff line
    ax[1,0].axvline(pred_cutoff, ls='--', color='w')

    ax[1,0].axis('scaled')

    ax[1,0].set_xlabel('Enformer prediction')
    ax[1,0].set_ylabel('Ground truth')
    ax[1,0].text(-0.05, 1.05, 'C',
               transform=ax[1,0].transAxes, ha='left', fontsize=20) 
    
    # ------------------------------------------------------------------ #
    # Lower Right: Violin — class by GT cutoff, Y = ground truth values
    # ------------------------------------------------------------------ #
    pred_class = (x >= pred_cutoff).astype(int)
    
    vp_ur = sns.violinplot(x = pred_class, y = df['Ground Truth'],ax=ax[1,1],inner="quart",color='gray')
    ax[1,1].axvline(0.5, ls='--', color='gray', alpha=0.7)
    ax[1,1].set_xticks([0,1])
    ax[1,1].set_xticklabels(['Off (pred $<$ cutoff)', 'On (pred $\\geq$ cutoff)'])
    ax[1,1].set_xlabel('Naive prediction')
    ax[1,1].set_ylabel('Ground Truth')
    # ax[1,1].set_title('pred-based classification')
    ax[1,1].text(-0.05, 1.05, 'D',
               transform=ax[1,1].transAxes, ha='left', fontsize=20)    

    if save_file is not None:
        fig.savefig(save_file,bbox_inches='tight')
    
def figure_S4(ground_truth, OR_gene_idx, cancer_CAGE, N, save_file=None):
    plt.rcParams['text.usetex'] = False
    fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(8,4),sharey=True)
    
    random_mean_expression = np.zeros((N,ground_truth.shape[1]))
    random_cov = np.zeros(N)
    for i in range(N):
        idx = np.random.permutation(len(ground_truth))[:len(OR_gene_idx)]
        random_mean_expression[i] = np.mean(ground_truth[idx],axis=0)
        random_cov[i] = np.mean(np.corrcoef(ground_truth[idx]))

    OR_mean_expression = np.mean(ground_truth[OR_gene_idx],axis=0)
    OR_cancer_mean_expression = np.mean(ground_truth[np.ix_(OR_gene_idx,cancer_CAGE)],axis=0)

    OR_mean_cov = np.mean(np.corrcoef(ground_truth[OR_gene_idx]))
    OR_cancer_mean_cov = np.mean(np.corrcoef(ground_truth[np.ix_(OR_gene_idx,cancer_CAGE)]))
        
    ax1.axvline(OR_mean_cov,ls='--',color='red',label='OR genes')
    ax1.axvline(OR_cancer_mean_cov,ls='--',color='darkred',label='OR genes (cancer tracks only)')
    sns.histplot(random_cov,color='gray',label='Random sample',ax=ax1)
    ax1.set_xlabel('Mean correlation across all pairs of 44 genes')
    ax1.legend(loc=(0.65,-0.45))
    
    ax2.axvline(np.mean(OR_mean_expression),ls='--',color='red',label='OR genes')
    ax2.axvline(np.mean(OR_cancer_mean_expression),ls='--',color='darkred',label='OR genes (cancer tracks only)')
    sns.histplot(np.mean(random_mean_expression,axis=1),color='gray',label='Random sample',ax=ax2)
    ax2.set_xlabel('Mean expression of 44 genes\naveraged across all tracks')
    
    if save_file is not None:
        fig.savefig(save_file,bbox_inches='tight')