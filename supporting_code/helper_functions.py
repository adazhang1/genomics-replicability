import numpy as np
import pandas as pd
import scipy
import tqdm
from sklearn.mixture import GaussianMixture
from collections import defaultdict

class PearsonR:
    def __init__(self,num_targets,summarize=True):
        self._summarize = summarize
        self._shape = (num_targets,)
        self._count = np.zeros(self._shape)
        
        self._product = np.zeros(self._shape)
        self._true_sum = np.zeros(self._shape)
        self._true_sumsq = np.zeros(self._shape)
        self._pred_sum = np.zeros(self._shape)
        self._pred_sumsq = np.zeros(self._shape)
    
    def update(self,y_true,y_pred):
        if len(y_true.shape) <= 2:
            reduce_axes = 0
        else:
            reduce_axes = (0,1)
            
        self._product += np.sum(y_true * y_pred,axis=reduce_axes)
        self._true_sum += np.sum(y_true,axis = reduce_axes)
        self._true_sumsq += np.sum(np.square(y_true),axis = reduce_axes)
        self._pred_sum += np.sum(y_pred,axis = reduce_axes)
        self._pred_sumsq += np.sum(np.square(y_pred),axis = reduce_axes)
        
        self._count += np.sum(np.ones_like(y_true),reduce_axes)
        
    def compute(self):
        """
        If no calls to ``update()`` are made before ``compute()`` is called,
        the function throws a warning and returns 0.0.
        """
        if not self._count.any():
            print("Warning for PearsonR object: No calls to update() have been made - returning 0.0")
            return np.zeros(self._shape)
        
        true_mean = np.divide(self._true_sum, self._count)
        pred_mean = np.divide(self._pred_sum, self._count)
        
        covariance = self._product - self._count * true_mean * pred_mean
        
        true_var = self._true_sumsq - self._count*np.square(true_mean)
        pred_var = self._pred_sumsq - self._count*np.square(pred_mean)
        
        pred_var = np.where(pred_var > 1e-12, pred_var, np.inf)
        
        correlation = np.divide(covariance, np.sqrt(true_var * pred_var))
        
        if self._summarize:
            return np.mean(correlation)
        else:
            return correlation

def process_gtf_file(filepath):
    column_names = ["seqnames", "source", "feature", "start", "end", "score", "strand", "frame", "attributes"]
    df = pd.read_csv(
            filepath,
            sep='\t',  # GTF files are typically tab-separated
            comment="#", # GTF files often have comment lines starting with '#'
            header=None, # GTF files usually don't have a header row
            names=column_names # Assign column names based on GTF specification
        )

    attributes = df['attributes'].str.split(';')
    parsed_attributes = attributes.apply(parse_attribute_string)
    df = df.join(pd.DataFrame(parsed_attributes.tolist()))
    
    # extract protein coding genes, only keep the transcripts to get multiple TSS for each gene.
    # Note that rows of type "gene" summarize info from multiple transcript rows for that gene.
    protein_coding_transcripts = (df['gene_type']=='protein_coding') & (df['feature']=='transcript')
    
    return df[protein_coding_transcripts]
    
def locate_TSS_in_Basenji_tracks(promoter_df,basenji_sequences_path,partition):
    # promoter_df dataframe that must contain the following columns:
    #    seqnames - the name of the chromosome
    #    start - start position of the gene
    #    end - end position of the gene
    #    strand - defined as + (forward) or - (reverse)
    #    gene_id - 
    # basenji_sequences_path points to the file that contains the loci of each basenji sample
    # partition is a string that represents which partition (train, valid, test) should be analyzed
    
    # grab TSS based on whether the TSS is on the + or - strand
    promoter_df.loc[promoter_df['strand'] == '+','tss'] = promoter_df.loc[promoter_df['strand'] == '+']['start']
    promoter_df.loc[promoter_df['strand'] == '-','tss'] = promoter_df.loc[promoter_df['strand'] == '-']['end']

    seq_ind = 0
    gene_inds = dict()
    gi = 0
    tss = []
    first = True
    with open(basenji_sequences_path) as f:
        for i, line in enumerate(tqdm.tqdm(f,total=38170)):
            chrom, start_ind, end_ind, p = line.strip('\n').split('\t')
    
            if p != partition:
                # training or validation sample, so skip
                continue 
    
            # setting up datasets for this sample
            tss_bins = []
            all_tss_idx = set()
            
            # adjusting start/end inds by the 64 bins (128bp per bin) on each side that were cropped during prediction
            start = int(start_ind) + 128*64
            end = int(end_ind) - 128*64
            
            # filtering the promoter df to this chromosome, and then to only the genes that have a tss within the start/end indices
            df = promoter_df[promoter_df.seqnames == chrom]
            genes = df.loc[(df.tss > start) & (df.tss < end)]
    
            # process each TSS in this sequence
            for tss_idx,ensg in zip(genes.tss,genes.gene_id):
                # convert gene index into bin index for this TSS
                tss_bin = int((tss_idx-start)//128)
    
                # if we haven't encountered this ENSG yet, add it to the ENSG -> index conversion dict
                if ensg not in gene_inds:
                    gene_inds[ensg] = gi
                    gi += 1
                
                if tss_bin > 895:
                    # somehow we're out of bounds, but this should've been caught when df was filtered to within this range and sample start/end were cropped
                    print(start,end,genes)
    
                # Check if we have overlapping TSS from different genes
                if tss_idx in all_tss_idx and (tss_bin,ensg) not in tss_bins:
                    if first:
                        print('Overlapping genes:')
                        first = False
                    # same TSS, different gene, print out overlapping genes
                    print('\t',chrom,start,end,tss_idx,tss_bin,ensg,[x for x in tss_bins if x[0]==tss_bin])
                else: # add this TSS to the set
                    all_tss_idx.add(tss_idx)
    
                # add bin and +/- 1 bin on either side to our set (as long as these bins are not yet in the set)
                for j in range(tss_bin-1,tss_bin+2):
                    if j < 0 or j > 895:
                        continue
                        
                    if (j,ensg) not in tss_bins:
                        tss_bins.append((j,ensg))
    
            tss.append([seq_ind,chrom,start,end,tss_bins])
            seq_ind += 1

    return tss, gene_inds

def extract_Basenji_CAGE_cols(basenji_targets_path):
    CAGE_cols = []
    with open(basenji_targets_path) as f:
        for line in f:
            l = line.split('\t')
            desc = l[-1]
            i = l[0]
            if desc[:4] == 'CAGE':
                CAGE_cols.append(int(i))
    return np.array(CAGE_cols)

def extract_gene_expression(shape,gene_inds,tss,tracks):
    gene_exp = np.zeros(shape)
    
    # for each model output
    for seq_ind in range(len(tracks)):
        for bin_ind, ensg in tss[seq_ind][-1]:
            gi = gene_inds[ensg]
            
            gene_exp[gi] += tracks[seq_ind][bin_ind]

    return gene_exp

def normalize(gt,pred,axis=0):
    x = np.log10(1 + gt)
    y = np.log10(1 + pred)
    
    gt_mean = np.mean(x,axis=axis)
    gt_std = np.std(x,axis=axis)
    
    return (x - gt_mean)/gt_std, (y - gt_mean)/gt_std
    
def estimate_distribution_density(x,y,model):
    x = x.flatten()
    y = y.flatten()

    data , x_e, y_e = np.histogram2d( x, y, bins = [30,30], density = True )
    z = scipy.interpolate.interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)
    
    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0
    
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    
    df = pd.DataFrame({'Prediction':x,'Ground Truth':y, 'Density':z,'Model':[model]*len(x)})

    return df

def fast_pearson(x, y):
    x = np.squeeze(x)
    y = np.squeeze(y)
    x = x - x.mean()
    y = y - y.mean()
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    
def compute_per_track_corr(x,y):
    # computing correlation per target track
    # array shapes are [# samples, # loci, # tracks]
    num_tracks = x.shape[2]
    x = x.transpose((2,0,1)).reshape((num_tracks,-1))
    y = y.transpose((2,0,1)).reshape((num_tracks,-1))
    # now shape is [# tracks, # samples x # loci]

    # checking where min = max in x or y
    invalid_inds = np.equal(np.max(x,axis=1),np.min(x,axis=1)) | np.equal(np.max(y,axis=1),np.min(y,axis=1))
    
    corr = np.zeros(num_tracks)
    for i in tqdm.tqdm(range(num_tracks)):
        corr[i] = fast_pearson(x[i],y[i])
        
    corr[invalid_inds] = 0

    return corr

def compute_density(x,y):
    data , x_e, y_e = np.histogram2d( x, y, bins = [30,30], density = True )
    z = scipy.interpolate.interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)
    
    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    return x,y,z

def compute_running_corr(x,y):
    # TODO: FIX FOR OFF GENES
    corr = PearsonR(1)
    corr.update(y[:2],x[:2])
    running_corr = []
    cut_off = [y[1]]
    num_samples = []
    for i in range(2,len(y)):
        if y[i] != cut_off[-1]:
            running_corr.append(corr.compute())
            cut_off.append(y[i])
            num_samples.append(i)
        corr.update(y[i],x[i])
    running_corr.append(corr.compute())
    num_samples.append(i)

    return running_corr,num_samples,cut_off

def compute_running_random_corr(N,x,y):
    running_corr = []
    i = 0
    corr = PearsonR(1)
    for n in N:
        n = int(n)
        if n == 0:
            running_corr.append(0)
            continue

        while i < n:
            corr.update(x[i],y[i])
            i+=1

        running_corr.append(corr.compute())

    return running_corr
    
def generate_running_correlation_dataframes(model,ground_truth,pred,cutoff_interp):
    df_ON = pd.DataFrame()
    df_OFF = pd.DataFrame()
    
    model_name = [model]*len(cutoff_interp)
    num_genes = ground_truth.shape[0]
    
    for track_ind in tqdm.tqdm(range(ground_truth.shape[1])):
        
        gt_track = ground_truth[:,track_ind]
        pred_track = pred[:,track_ind]

        index_array = np.argsort(gt_track)
        gt_track = gt_track[index_array]
        pred_track = pred_track[index_array]

        # getting running correlation of OFF genes
        running_corr_off,num_samples_off,cutoff_off = compute_running_corr(pred_track,gt_track)

        running_corr_interp = np.interp(cutoff_interp,cutoff_off,running_corr_off)
        cutoff_sample_interp_off = np.interp(cutoff_interp,cutoff_off,num_samples_off)

        df = pd.DataFrame({'Cutoff':cutoff_interp,
                           'Correlation':running_corr_interp,
                           '# samples':cutoff_sample_interp_off,
                           '% of genes':cutoff_sample_interp_off/num_genes*100,
                           'Track': [track_ind]*len(cutoff_interp),
                           'Model':model_name})
        df_OFF = pd.concat([df,df_OFF], ignore_index = True)

        # getting running correlation of ON genes
        running_corr_on,num_samples_on,cutoff_on = compute_running_corr(np.flip(pred_track),np.flip(gt_track))

        running_corr_interp = np.interp(cutoff_interp,cutoff_on[::-1],running_corr_on[::-1])
        cutoff_sample_interp_on = np.interp(cutoff_interp,cutoff_on[::-1],num_samples_on[::-1])
        df = pd.DataFrame({'Cutoff':cutoff_interp,
                           'Correlation':running_corr_interp,
                           '# samples':cutoff_sample_interp_on,
                           '% of genes':cutoff_sample_interp_on/num_genes*100,
                           'Track': [track_ind]*len(cutoff_interp),
                           'Model':model_name})
        df_ON = pd.concat([df,df_ON], ignore_index = True)
    
    return df_ON, df_OFF

def generate_randomly_sampled_running_correlation_dataframes(model,ground_truth,pred,cutoff_interp,model_ON,model_OFF):
    df_ON = pd.DataFrame()
    df_OFF = pd.DataFrame()
    
    model_name_random = [f'{model} random sample']*len(cutoff_interp)
    
    for track_ind in tqdm.tqdm(range(ground_truth.shape[1])):
        gt_track = ground_truth[:,track_ind]
        pred_track = pred[:,track_ind]
        
        index_array = np.random.permutation(len(gt_track))
        gt_track = gt_track[index_array]
        pred_track = pred_track[index_array]

        # get random sample by expression cutoff (OFF)
        track_rows = model_OFF['Track'] == track_ind
        sample_interp_off = model_OFF[track_rows]['# samples']
        random_corr = compute_running_random_corr(sample_interp_off,gt_track,pred_track)
        df = pd.DataFrame({'Cutoff':cutoff_interp,
                           'Correlation':random_corr,
                           '# samples':sample_interp_off,
                           'Track': [track_ind]*len(cutoff_interp),
                           'Model':model_name_random})
        df_OFF = pd.concat([df,df_OFF], ignore_index = True)
        
        # get random sample by expression cutoff (ON)
        track_rows = model_ON['Track'] == track_ind
        sample_interp_on = model_ON[track_rows]['# samples']
        random_corr = compute_running_random_corr(sample_interp_on[::-1],gt_track,pred_track)
        df = pd.DataFrame({'Cutoff':cutoff_interp,
                           'Correlation':random_corr[::-1],
                           '# samples':sample_interp_on,
                           'Track': [track_ind]*len(cutoff_interp),
                           'Model':model_name_random})
        df_ON = pd.concat([df,df_ON], ignore_index = True)

    return df_ON, df_OFF

def compute_GMM_dist(ground_truth, disable_tqdm=False):
    distances = []
    for track_ind in tqdm.tqdm(range(ground_truth.shape[1]), disable=disable_tqdm):
        x = ground_truth[:,track_ind]

        estimator = GaussianMixture(n_components=2,covariance_type='full',random_state=0)
        estimator.fit(x.reshape(-1, 1))

        distances.append(np.linalg.norm(estimator.means_[0]-estimator.means_[1]))
        
    return distances
    
def generate_normalized_absolute_error_medians(ground_truth, predictions, tissue_df, gene_ENSG):
    abs_err = dict()
    for model in predictions.keys():
        # absolute error between model and ground truth
        err = np.abs(predictions[model] - ground_truth)

        # normalizing error by gene expression magnitude within each track
        abs_err[f'normalized_{model}'] = np.zeros_like(ground_truth)

        for i in range(len(tissue_df)):

            # grabbing error and ground truth for each track
            e = err[:,i]
            gt = ground_truth[:,i]

            # linear fit of ground truth value to error
            m,b = np.polyfit(gt,e,1)

            # saving residuals of linear fit
            abs_err[f'normalized_{model}'][:,i] = e - (m*gt + b)
    
    # NOTE: rows are genes, columns are CAGE tracks.
    
    # saving median error of cancerous tissues versus non-cancerous tissues for each gene
    for model in abs_err.keys():
        temp_df = pd.DataFrame()
        temp_df['ENSG'] = gene_ENSG
        
        cancer_mask = (tissue_df['Cancer'] == 'Yes').to_numpy()
        temp_df['Cancer'] = np.median(abs_err[model][:,cancer_mask],axis=1)
        temp_df['Not Cancer'] = np.median(abs_err[model][:,~cancer_mask],axis=1)
    
        temp_df.to_csv(f'./GSEA_tissue_cancer_error/data/{model}_cancer.csv')


    # saving median error of each tissue type for each gene (combining cancer and non-cancer)
    for model in abs_err.keys():
        temp_df = pd.DataFrame()
        temp_df['ENSG'] = gene_ENSG
    
        for tissue in tissue_df['Tissue'].unique():
            tissue_mask = (tissue_df['Tissue'] == tissue).to_numpy()
            temp_df[tissue] = np.median(abs_err[model][:,tissue_mask],axis=1)
    
        temp_df.to_csv(f'./GSEA_tissue_cancer_error/data/{model}_tissue.csv')

    # saving median error of each tissue type for each gene, separated by cancer v. non-cancer
    for model in abs_err.keys():
        temp_df = pd.DataFrame()
        temp_df['ENSG'] = gene_ENSG
        
        cancer_mask = (tissue_df['Cancer'] == 'Yes').to_numpy()
        for tissue in tissue_df['Tissue'].unique():
            tissue_mask = (tissue_df['Tissue'] == tissue).to_numpy()
            temp_df[f'{tissue}_cancer'] = np.median(abs_err[model][:,tissue_mask&cancer_mask],axis=1)
            temp_df[f'{tissue}_noncancer'] = np.median(abs_err[model][:,tissue_mask&~cancer_mask],axis=1)
    
        temp_df.to_csv(f'./GSEA_tissue_cancer_error/data/{model}_tissue_and_cancer.csv')

def generate_bimodal_gaussian_from_data(x,y,corr,cutoff):
    x_on = x[x >= cutoff]
    y_on = y[y >= cutoff]

    x_off = x[x < cutoff]
    y_off = y[y < cutoff]

    samples_on = generate_unimodal_gaussian_from_data(x_on,y_on,corr)
    samples_off = generate_unimodal_gaussian_from_data(x_off,y_off,corr)

    samples = np.concatenate((samples_on,samples_off))
    
    return samples

def generate_unimodal_gaussian_from_data(x,y,corr):
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    x_var = np.var(x)
    y_var = np.var(y)

    cov = corr*np.sqrt(x_var*y_var)

    samples = np.random.multivariate_normal([x_mean,y_mean],[[x_var,cov],[cov,y_var]],len(x))

    return samples
    
def generate_synthetic_data(gene_exp_gt,gene_exp_pred,cutoff):
    unimodal = {'x':np.zeros_like(gene_exp_gt),'y':np.zeros_like(gene_exp_gt)}
    bimodal_track_corr = {'x':np.zeros_like(gene_exp_gt),'y':np.zeros_like(gene_exp_gt)}
    bimodal_zero_corr = {'x':np.zeros_like(gene_exp_gt),'y':np.zeros_like(gene_exp_gt)}

    for track_ind in tqdm.tqdm(range(gene_exp_gt.shape[1])):
        gt_track = gene_exp_gt[:,track_ind]
        pred_track = gene_exp_pred[:,track_ind]

        track_corr = np.corrcoef(gt_track,pred_track)[0,1]

        # generate unimodal data
        samples = generate_unimodal_gaussian_from_data(pred_track,gt_track,track_corr)
        unimodal['x'][:,track_ind] = samples[:,0]
        unimodal['y'][:,track_ind] = samples[:,1]

        # generate bimodal data with track corr
        samples = generate_bimodal_gaussian_from_data(pred_track,gt_track,track_corr,cutoff)
        bimodal_track_corr['x'][:,track_ind] = samples[:,0]
        bimodal_track_corr['y'][:,track_ind] = samples[:,1]
        
        # generate bimodal data with zero corr
        samples = generate_bimodal_gaussian_from_data(pred_track,gt_track,0,cutoff)
        bimodal_zero_corr['x'][:,track_ind] = samples[:,0]
        bimodal_zero_corr['y'][:,track_ind] = samples[:,1]

    return {'Unimodal': unimodal, 'Bimodal - overall track correlation': bimodal_track_corr, 'Bimodal - zero correlation': bimodal_zero_corr}

def compute_naive_corr(model,ground_truth, prediction, gt_cutoff,pred_cutoff):
    
    pred = prediction.flatten()
    gt = ground_truth.flatten()

    # generate naive prediction based on ground truth or model prediction
    gt_naive = gt > gt_cutoff
    pred_naive = pred > pred_cutoff

    # store model performance
    out = {
        'Source':['model'],
        'Correlation':[scipy.stats.pearsonr(gt, pred)[0]],
    }
        
    for naive,source in zip([gt_naive,pred_naive],['ground truth','model prediction']):
        out['Source'].append(source)
        out['Correlation'].append(scipy.stats.pearsonr(gt,naive)[0])
        
    out['Model'] = [model]*3

    return out
