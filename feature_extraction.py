import torch
import re
from torch import nn

_restypes = ["A","R","N","D","C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",]
_restypes_with_x = _restypes + ["X"]
_restypes_with_x_and_gap = _restypes_with_x + ["-"]

restype_order_with_x = {}
restype_order_with_x_and_gap = {}


# Initialize the variables above as dicts mapping the              
#   residues to their corresponding index in the list.                   

# Replace "pass" statement with your code
for i, aa in enumerate(_restypes_with_x):
    restype_order_with_x[aa] = i
for i, aa in enumerate(_restypes_with_x_and_gap):
    restype_order_with_x_and_gap[aa] = i



def load_a3m_file(file_name: str):
    """
    Loads an A3M (multiple sequence alignment) file and extracts the raw amino acid sequences.

    Args:
        file_name: Path to the A3M file.

    Returns:
        A list of strings where each string represents an individual protein sequence from the input MSA.
    """

    seqs = None

    with open(file_name, 'r') as f:
        lines = f.readlines()

    arrow_line_indices = [i for i,l in enumerate(lines) if l.startswith('>')]
    seqs = [lines[i+1].strip() for i in arrow_line_indices]

    return seqs



def onehot_encode_aa_type(seq, include_gap_token=False):
    """
    Converts a protein sequence into one-hot encoding. X represents an unkown amino acid.

    Args:
        seq:  A string representing the amino acid sequence using single-letter codes.
        include_gap_token: If True, includes an extra token ('-') in the encoding to 
                           represent gaps.

    Returns: 
        A PyTorch tensor of shape (N_res, 22) if `include_gap_token` is True, 
        or shape (N_res, 21) otherwise.  Here, N_res is the length of the sequence.
    """
    restype_order = restype_order_with_x if not include_gap_token else restype_order_with_x_and_gap
    encoding = None

    n = 22 if include_gap_token else 21
    indices = [restype_order[aa] for aa in seq]
    encoding = nn.functional.one_hot(torch.tensor(indices).long(), num_classes = n)


    return encoding



def initial_data_from_seqs(seqs):
    """
    Processes raw sequences from an A3M file to extract initial feature representations.

    Args:
        seqs: A list of amino acid sequences loaded from the A3M file. 
              Sequences are represented with single-letter amino acid codes.
              Lowercase letters represent deletions.

    Returns:
        A dictionary containing:
            * msa_aatype: A PyTorch tensor of one-hot encoded amino acid sequences
                  of shape (N_seq, N_res, 22), where N_seq is the number of unique 
                  sequences (with deletions removed) and N_res is the length of the sequences. 
                  The dimension 22 corresponds to the 20 amino acids, an unknown amino acid 
                  token, and a gap token. 
            * msa_deletion_count: A tensor of shape (N_seq, N_res) where 
                  each element represents the number of deletions occurring before 
                  the corresponding residue in the MSA.
            * aa_distribution: A tensor of shape (N_res, 22) containing the 
                  overall amino acid distribution at each residue position 
                  across the MSA.  
    """

    unique_seqs = None
    deletion_count_matrix = None
    aa_distribution = None
    unique_seqs = []
    deletion_count_matrix = []

    for seq in seqs:
        deletions_list = []
        current_count = 0
        for aa in seq:
            if aa.islower():
                current_count += 1
            else:
                deletions_list.append(current_count)
                current_count = 0

        seq_no_dels = re.sub("[a-z]", "", seq)
        if seq_no_dels not in unique_seqs:
            unique_seqs.append(seq_no_dels)
            deletion_count_matrix.append(deletions_list)

    deletion_count_matrix = torch.tensor(deletion_count_matrix).float() #(N_seq, N_res)

    tensors = []
    for seq in unique_seqs:
        tensors.append(onehot_encode_aa_type(seq, include_gap_token=True))
    unique_seqs = torch.stack(tensors, dim = 0).float() # (N_seq, N_res, 22)

    aa_distribution = torch.mean(unique_seqs, 0)  # (N_res, 22)
    


    return { 'msa_aatype': unique_seqs, 'msa_deletion_count': deletion_count_matrix, 'aa_distribution': aa_distribution}

def select_cluster_centers(features, max_msa_clusters=512, seed=None):
    """
    Selects representative sequences as cluster centers from the MSA to  
    reduce redundancy.

    Args:
        features: A dictionary containing feature representations of the MSA.
        max_msa_clusters: The maximum number of cluster centers to select.
        seed: An optional integer seed for the random number generator. 
              Use this to ensure reproducibility.

    Modifies:
        The 'features' dictionary in-place by:
            * Updating the 'msa_aatype' and 'msa_deletion_count' features to contain 
              data for the cluster centers only.  
            * Adding 'extra_msa_aatype' and 'extra_msa_deletion_count' features
              to hold the data for the remaining (non-center) sequences. 
    """

    N_seq, N_res = features['msa_aatype'].shape[:2]
    MSA_FEATURE_NAMES = ['msa_aatype', 'msa_deletion_count']
    max_msa_clusters = min(max_msa_clusters, N_seq)

    gen = None
    if seed is not None:
        gen = torch.Generator(features['msa_aatype'].device)
        gen.manual_seed(seed)


    permuted_indices = torch.cat([torch.tensor([0]), torch.randperm(N_seq - 1, generator = gen) + 1], dim = 0)
    cluster_indices = permuted_indices[:max_msa_clusters]
    extra_msa_indices = permuted_indices[max_msa_clusters:]

    msa_aatype = features['msa_aatype'][cluster_indices, :, :]
    msa_deletion_count = features['msa_deletion_count'][cluster_indices, :]

    extra_msa_aatype = features['msa_aatype'][extra_msa_indices, :, :]
    extra_msa_deletion_count = features['msa_deletion_count'][extra_msa_indices, :]

    features['msa_aatype'] = msa_aatype # (N_clust, N_res, 22)
    features['msa_deletion_count'] = msa_deletion_count # (N_clust, N_res)
    features['extra_msa_aatype'] = extra_msa_aatype  # (N_extra, N_res, 22)
    features['extra_msa_deletion_count'] = extra_msa_deletion_count # (N_extra, N_res)


    return features

def mask_cluster_centers(features, mask_probability=0.15, seed=None):
    """
    Introduces random masking in the cluster center sequences for data augmentation.

    This function modifies the 'msa_aatype' feature within the 'features' dictionary to improve 
    model robustness in the presence of noisy or missing input data.  Masking is inspired by 
    the AlphaFold architecture.

    Args:
        features: A dictionary containing feature representations of the MSA. It is assumed
                  that cluster centers have already been selected.
        mask_probability: The probability of masking out an individual amino acid 
                          in a cluster center sequence.
        seed: An optional integer seed for the random number generator. 
              Use this to ensure reproducibility.

    Modifies:
        The 'features' dictionary in-place by:
            * Updating the 'msa_aatype' feature with masked-out tokens as well as possible 
              replacements based on defined probabilities. 
            * Creating a copy of the original 'msa_aatype' feature with the key 'true_msa_aatype'. 
    """

    N_clust, N_res = features['msa_aatype'].shape[:2]
    N_aa_categories = 23 # 20 Amino Acids, Unknown AA, Gap, masked_msa_token
    odds = {
        'uniform_replacement': 0.1,
        'replacement_from_distribution': 0.1,
        'no_replacement': 0.1,
        'masked_out': 0.7,
    }
    gen = None
    if seed is not None:
        gen = torch.Generator(features['msa_aatype'].device)
        gen.manual_seed(seed)
        torch.manual_seed(seed)


    mask = torch.rand((N_clust, N_res), generator = gen) < mask_probability

    # uniform_replacement has shape (22,)
    uniform_replacement = torch.tensor(([1/20*odds['uniform_replacement']]*20 + [0,0]))
    # replacement_from_distribution has shape (N_res, 22)
    replacement_from_distribution = features['aa_distribution'] * odds['replacement_from_distribution']
    # no_replacement has shape (N_clust, N_res, 22)
    no_replacement = features['msa_aatype'] * odds['no_replacement']
    # masked_out has shape (N_clust, N_res, 1)
    masked_out = torch.ones((N_clust, N_res, 1)) * odds['masked_out']

    categories_with_mask_token = torch.cat([uniform_replacement + replacement_from_distribution + no_replacement, masked_out], dim = -1)
    categories_with_mask_token = categories_with_mask_token.reshape(-1, N_aa_categories)

    sampling_dist = torch.distributions.Categorical(probs = categories_with_mask_token)
    samples = sampling_dist.sample()
    samples = nn.functional.one_hot(samples, num_classes = N_aa_categories).view(N_clust, N_res, N_aa_categories).float()

    features['true_msa_aatype'] = torch.clone(features['msa_aatype']) # (N_clust, N_res, 22)
    features['msa_aatype'] = torch.cat([features['msa_aatype'], torch.zeros((N_clust, N_res, 1))], dim = -1) # (N_clust, N_res, 23)
    features['msa_aatype'][mask] = samples[mask]  # (N_clust, N_res, 23)

    return features

def cluster_assignment(features):
    """
    Assigns sequences in the extra MSA to their closest cluster centers based on Hamming distance.

    Args:
        features: A dictionary containing feature representations of the MSA. 
                  It is assumed that cluster centers have already been selected.

    Returns:
        The updated 'features' dictionary with the following additions:
            * cluster_assignment:  A tensor of shape (N_extra,) containing the indices 
                                  of the assigned cluster centers for each extra sequence.
            * cluster_assignment_counts: A tensor of shape (N_clust,)  where each element indicates 
                                        the number of extra sequences assigned to a cluster center 
                                        (excluding the cluster center itself).
    """
    
    N_clust, N_res = features['msa_aatype'].shape[:2]
    N_extra = features['extra_msa_aatype'].shape[0]

    msa_aatype_no_gap_no_masked = features['msa_aatype'][:, :, :21] # Shape (N_clust, N_res, 21)
    extra_msa_aatype_no_gap_no_masked = features['extra_msa_aatype'][:, :, :21] # Shape (N_extra, N_res, 21)

    # works but allocates too much memory
    # msa_aatype_no_gap_no_masked = msa_aatype_no_gap_no_masked.unsqueeze(1).broadcast_to((N_clust, N_extra, N_res, 21)) # Shape (N_clust, N_extra, N_res, 21)
    # extra_msa_aatype_no_gap_no_masked = extra_msa_aatype_no_gap_no_masked.unsqueeze(0).broadcast_to((N_clust, N_extra, N_res, 21)) # Shape (N_clust, N_extra, N_res, 21)
    # agreement = (msa_aatype_no_gap_no_masked * extra_msa_aatype_no_gap_no_masked).float().sum(dim = (-1, -2))

    agreement = torch.einsum('cra,era->ce', msa_aatype_no_gap_no_masked, extra_msa_aatype_no_gap_no_masked)

    features['cluster_assignment'] = torch.argmax(agreement, dim = 0)

    features['cluster_assignment_counts'] = torch.bincount(features['cluster_assignment'], minlength = N_clust) 

    return features

def cluster_average(feature, extra_feature, cluster_assignment, cluster_assignment_count):
    """
    Calculates the average representation of each cluster center by aggregating features 
    from the assigned extra sequences.

    Args:
        feature: A tensor containing feature representations for the cluster centers.
                 Shape: (N_clust, N_res, *)
        extra_feature: A tensor containing feature representations for extra sequences.
                       Shape: (N_extra, N_res, *).  The trailing dimensions (*) must 
                       be smaller or equal to those of the 'feature' tensor.
        cluster_assignment: A tensor indicating the cluster assignment of each extra sequence.
                            Shape: (N_extra,)
        cluster_assignment_count: A tensor containing the number of extra 
                                 sequences assigned to each cluster center.
                                 Shape: (N_clust,)

    Returns:
        A tensor containing the average feature representation for each cluster. 
        Shape: (N_clust, N_res, *) 
    """
    N_clust, N_res = feature.shape[:2]
    N_extra = extra_feature.shape[0]


    cluster_assignment = cluster_assignment[(...,) + (None,) * (extra_feature.ndim - 1)].broadcast_to((extra_feature.size()))
    # summed_features_per_cluster is (N_cluster, N_res, *)
    summed_features_per_cluster = torch.scatter_add(feature, 0, cluster_assignment, extra_feature)
    # reshape cluster_assignment_count to (N_clust, 1, 1) (as many trailing dims added as necessary)
    cluster_assignment_count = cluster_assignment_count[(...,) + (None,) * (feature.ndim - 1)]
    cluster_average = summed_features_per_cluster / (cluster_assignment_count+1)


    return cluster_average



def summarize_clusters(features):
    """
    Calculates cluster summaries by applying cluster averaging to the MSA amino acid 
    representations and deletion counts.

    Args:
        features: A dictionary containing feature representations of the MSA.

    Modifies:
        The 'features' dictionary in-place by adding the following:
            * cluster_deletion_mean: Average deletion counts for each cluster center, 
                                     scaled for numerical stability.
            * cluster_profile: Average amino acid representations for each cluster center.
    """

    N_clust, N_res = features['msa_aatype'].shape[:2]
    N_extra = features['extra_msa_aatype'].shape[0]

    cluster_assignment = features['cluster_assignment'] # (N_extra,)
    cluster_assignment_counts = features['cluster_assignment_counts'] # (N_clust,)

    cluster_msa_deletion_count = features['msa_deletion_count'] # (N_clust, N_res)
    extra_msa_deletion_count = features['extra_msa_deletion_count'] # (N_extra, N_res)

    # (N_clust, N_res)
    cluster_average_deletion_counts = cluster_average(cluster_msa_deletion_count, extra_msa_deletion_count, cluster_assignment, cluster_assignment_counts)
    cluster_deletion_mean = 2 / torch.pi * torch.arctan(cluster_average_deletion_counts / 3) # (N_clust, N_res)

    features['cluster_deletion_mean'] = cluster_deletion_mean # (N_clust, N_res)

    cluster_msa_aatype = features['msa_aatype'] # (N_clust, N_res, 23)
    extra_msa_aatype = features['extra_msa_aatype'] # (N_extra, N_res, 22)
    extra_msa_aatype = torch.cat([features['extra_msa_aatype'], torch.zeros(N_extra, N_res, 1)], dim = -1) # (N_extra, N_res, 23)

    # (N_clust, N_res, 23)
    cluster_profile = cluster_average(cluster_msa_aatype, extra_msa_aatype, cluster_assignment, cluster_assignment_counts)

    features['cluster_profile'] = cluster_profile # (N_clust, N_res, 23)

    return features

def crop_extra_msa(features, max_extra_msa_count=5120, seed=None):
    """
    Reduces the number of extra sequences in the MSA to a fixed size for computational efficiency.

    Args:
        features: A dictionary containing feature representations of the MSA.
        max_extra_msa_count: The maximum number of extra sequences to retain.
        seed: An optional integer seed for the random number generator. 
              Use this to ensure reproducibility.

    Modifies:
        The  'features' dictionary in-place by cropping the following keys to include
        only the first 'max_extra_msa_count' sequences:
            * Any key starting with 'extra_' 
    """

    N_extra = features['extra_msa_aatype'].shape[0]
    gen = None
    if seed is not None:
        gen = torch.Generator(features['extra_msa_aatype'].device)
        gen.manual_seed(seed)

    max_extra_msa_count = min(max_extra_msa_count, N_extra)

    extra_seq_indices = torch.randperm(N_extra, generator = gen)[:max_extra_msa_count]
    for key, value in features.items():
        if key.startswith('extra_'):
            features[key] = features[key][extra_seq_indices]


    return features

def calculate_msa_feat(features):
    """
    Prepares the final MSA feature representation for protein structure prediction.

    Args:
        features: A dictionary containing feature representations of the MSA.

    Returns:
        A tensor of shape (N_clust, N_res, 49) representing the final MSA features,
        formed by concatenating processed cluster information and deletion-related values. 
    """
    
    N_clust, N_res = features['msa_aatype'].shape[:2]
    msa_feat = None

    cluster_msa = features['msa_aatype'] # (N_clust, N_res, 23)
    msa_deletion_count = features['msa_deletion_count'] # (N_clust, N_res)
    cluster_deletion_mean = features['cluster_deletion_mean'].unsqueeze(-1) # (N_clust, N_res, 1)
    cluster_profile = features['cluster_profile'] # (N_clust, N_res, 23))

    cluster_has_deletion = (msa_deletion_count != 0).float().unsqueeze(-1) # (N_clust, N_res, 1)
    cluster_deletion_value = 2 / torch.pi * torch.arctan(msa_deletion_count / 3).unsqueeze(-1) # (N_clust, N_res, 1)

    msa_feat = torch.cat([cluster_msa, cluster_has_deletion, cluster_deletion_value, cluster_profile, cluster_deletion_mean], dim = -1)

    ##########################################################################
    # END OF YOUR CODE                                                       #
    ##########################################################################

    return msa_feat

def calculate_extra_msa_feat(features):
    """
    Prepares the extra MSA feature representation for protein structure prediction. 
    This function is similar to 'calculate_msa_feat' but operates on  extra MSA sequences
    and includes padding of extra_msa_aatype to match the shape of msa_aatype. 

    Args:
        features: A dictionary containing feature representations of the MSA.

    Returns:
        A tensor of shape (N_extra, N_res, 25) representing the final extra MSA features.
    """

    N_extra, N_res = features['extra_msa_aatype'].shape[:2]
    extra_msa_feat = None

    extra_msa_aatype = features['extra_msa_aatype'] # (N_extra, N_res, 22)
    extra_msa_aatype = torch.cat([extra_msa_aatype, torch.zeros((N_extra, N_res, 1))], dim = -1)
    extra_msa_deletion_count = features['extra_msa_deletion_count'] # (N_extra, N_res)

    extra_msa_has_deletion = (extra_msa_deletion_count != 0).float().unsqueeze(-1) # (N_extra, N_res, 1)
    extra_msa_deletion_value = 2 / torch.pi * torch.arctan(extra_msa_deletion_count / 3).unsqueeze(-1) # (N_extra, N_res, 1)

    extra_msa_feat = torch.cat([extra_msa_aatype, extra_msa_has_deletion, extra_msa_deletion_value], dim = -1)


    return extra_msa_feat



def create_features_from_a3m(file_name, seed=None):
    """
    Creates feature representations for an MSA from its A3M file.

    This function orchestrates a sequence of transformations on the raw MSA sequences to 
    produce features suitable for protein structure prediction.

    Args:
        file_name: Path to the A3M file containing the MSA sequences.

    Returns:
        A dictionary containing the following feature representations for the MSA:
           * msa_feat: A tensor containing the final MSA feature representation.
           * extra_msa_feat: A tensor containing the final extra MSA feature representation.
           * target_feat: A tensor containing a one-hot encoded representation of the 
                          target protein sequence (excluding gaps and masked tokens).
           * residue_index: A tensor containing the residue indices (0, 1, ..., N_res-1). 
    """

    msa_feat = None
    extra_msa_feat = None
    target_feat = None
    residue_index = None
    select_clusters_seed = None
    mask_clusters_seed = None
    crop_extra_seed = None
    if seed is not None:
        select_clusters_seed = seed
        mask_clusters_seed = seed+1
        crop_extra_seed = seed+2
        

    seqs = load_a3m_file(file_name)
    features = initial_data_from_seqs(seqs)
    features = select_cluster_centers(features, max_msa_clusters=512, seed=select_clusters_seed)
    features = mask_cluster_centers(features, mask_probability=0.15, seed=mask_clusters_seed)
    features = cluster_assignment(features)
    features = summarize_clusters(features)
    crop_extra_msa(features, max_extra_msa_count=5120, seed=crop_extra_seed)

    msa_feat = calculate_msa_feat(features) # (N_clust, N_res, 49)
    extra_msa_feat = calculate_extra_msa_feat(features) # (N_extra, N_res, 25)

    target_feat = onehot_encode_aa_type(seqs[0], include_gap_token=False).float() # (N_res, 21)
    residue_index = residue_index = torch.arange(len(seqs[0]))

    return {
        'msa_feat': msa_feat,
        'extra_msa_feat': extra_msa_feat,
        'target_feat': target_feat,
        'residue_index': residue_index
    }


    