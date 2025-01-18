def plot_price_token(price_data, token_data, bins, token_limit):
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # Time Series
    ax1.plot(price_data.T)
    ax1.axhline(token_limit, color='red', linewidth=1)
    ax1.axhline(-token_limit, color='red', linewidth=1)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Delta Consumption')
    ax1.set_title('Time Series')

    # Boxplot Time Series
    ax2.boxplot(np.concatenate(price_data), vert=True)
    ax2.axhline(token_limit, color='red', linewidth=1)
    ax2.axhline(-token_limit, color='red', linewidth=1)
    ax2.set_title('Boxplot Delta Consumption')
    ax2.set_ylabel('Delta Consumption')

    # Histogram
    ax3.hist(np.concatenate(token_data), bins=bins, edgecolor='blue', color='orange', density=True)
    ax3.set_title('Histogram: Electricity Data Delta Consumption')
    ax3.set_xlabel('Delta Consumption')
    ax3.set_ylabel('Frequency')

    # 99% Data Histogram
    ax4.hist(np.concatenate(token_data), bins=bins, edgecolor='orange', color='blue', density=True)
    ax4.set_xlim(np.quantile(np.concatenate(token_data), q=0.025),np.quantile(np.concatenate(token_data), q=0.975))
    #ax4.set_xlim(np.quantile(np.concatenate(token_data), q=0.005),np.quantile(np.concatenate(token_data), q=0.995))
    ax4.set_title('Histogram: 99% of Data')
    ax4.set_xlabel('Delta Consumption')
    ax4.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def supervised_wasserstein(real_data, supervised_data):
    n_timesteps = real_data.shape[1]
    
    # Calculate Wasserstein distance for each timestep
    wasserstein_supervised = []
    
    for t in range(n_timesteps):
        w_dist = wasserstein_distance(real_data[:, t], supervised_data[:, t])
        wasserstein_supervised.append(w_dist)

    # Create visualization
    fig = plt.figure(figsize=(12, 4))
    
    # 1. Main grid for plots
    gs = plt.GridSpec(1, 2)
    # Plot 1: Wasserstein distances over time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(wasserstein_supervised, label = "Supervised")
    ax1.set_title('Wasserstein Distance by Timestep')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Wasserstein Distance')
    ax1.legend()

    # Plot 2: Compare specific timesteps
    ax2 = fig.add_subplot(gs[0, 1])
    # Pick timestep with largest Wasserstein distance
    worst_timestep_supervised = np.argmax(wasserstein_supervised)

    # Get data for the specified timestep
    real_data = real_data[:, worst_timestep_supervised]
    generated_data = supervised_data[:, worst_timestep_supervised]

    # Calculate KDE
    kde_real = gaussian_kde(real_data)
    kde_generated = gaussian_kde(generated_data)

    # Create evaluation points
    x_eval = np.linspace(min(real_data.min(), generated_data.min()),
                        max(real_data.max(), generated_data.max()),
                        200)
    
    # Plot distributions
    ax2.plot(x_eval, kde_real(x_eval), label='Real', color='blue')
    ax2.plot(x_eval, kde_generated(x_eval), label='Supervised', color='red')
    ax2.set_title(f'Distribution Comparison at Timestep {worst_timestep_supervised}\n'
                f'(Largest Wasserstein Distance)')
    ax2.legend()

def plot_first_n_distributions(real_data, supervised_data, n_steps=10):
    # Create visualization
    plots_per_row = 5
    rows = (n_steps + plots_per_row - 1) // plots_per_row  # Calculate number of rows needed (ceiling division)
    
    fig = plt.figure(figsize=(30, 4 * rows))  # Adjusted figure size to account for 5 plots per row
    gs = plt.GridSpec(rows, plots_per_row)
    
    for t in range(n_steps):
        # Calculate subplot position
        row = t // plots_per_row
        col = t % plots_per_row
        
        # Create subplot
        ax = fig.add_subplot(gs[row, col])
        
        # Get data for the timestep
        real_timestep = real_data[:, t]
        generated_timestep = supervised_data[:, t]
        
        # Calculate KDE
        kde_real = gaussian_kde(real_timestep)
        kde_generated = gaussian_kde(generated_timestep)
        
        # Create evaluation points
        x_eval = np.linspace(min(real_timestep.min(), generated_timestep.min()),
                            max(real_timestep.max(), generated_timestep.max()),
                            200)
        
        # Plot distributions
        ax.plot(x_eval, kde_real(x_eval), label='Real', color='blue', linewidth=0.8)
        ax.plot(x_eval, kde_generated(x_eval), label='Supervised', color='red', linewidth=0.8)
        ax.set_title(f'Timestep {t}')
        ax.legend()
    
    plt.tight_layout()
    plt.show()

def plot_autocorrelations(X_train, supervised_data, nlags=10):
    plt.figure(figsize=(10, 6))
    
    # Real data autocorrelation
    acf_real = acf(X_train[:, 1:].flatten(), nlags=nlags)
    plt.plot(range(len(acf_real)), acf_real, linewidth=1.5, label='Real Data')
    
    # Supervised (generated) data autocorrelation
    acf_supervised = acf(supervised_data.flatten(), nlags=nlags)
    plt.plot(range(len(acf_supervised)), acf_supervised, linewidth=1.5, label='Generated Data')
    
    plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
    plt.title('Autocorrelation Comparison')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_losses(pg_loss, value_loss, entropy_loss):

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize = (18,3))
    axs = axs.flatten()
    
    axs[0].plot(pg_loss, color='blue')
    axs[0].set_title('Policy Loss')
    axs[0].set_xlabel('Model Update')
    axs[0].set_ylabel('Value')

    axs[1].plot(value_loss, color='blue')
    axs[1].set_title('Value Loss')
    axs[1].set_xlabel('Model Update')
    axs[1].set_ylabel('Value')

    axs[2].plot(entropy_loss, color='blue')
    axs[2].set_title('Entropy Loss')
    axs[2].set_xlabel('Model Update')
    axs[2].set_ylabel('Value')
    
    #plt.tight_layout()
    plt.show()

def plot_sequence_metrics(ep_rew, wasserstein, kl_div):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize = (18,3))
    axs = axs.flatten()
    
    axs[0].plot(ep_rew, color='blue')
    axs[0].set_title('Episode Reward')
    axs[0].set_ylabel('Reward')
    axs[0].set_xlabel('Model Update')

    axs[1].plot(wasserstein, color='blue')
    axs[1].set_title('Wasserstein Distance')
    axs[1].set_ylabel('WS Distance')
    axs[1].set_xlabel('Model Update')

    axs[2].plot(kl_div, color='blue')
    axs[2].set_title('KL Div. real vs. sim sequences')
    axs[2].set_ylabel('KL Divergence')
    axs[2].set_xlabel('Model Update')

    plt.show()

def plot_discriminator(disc_loss, disc_acc, disc_acc_diff):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize = (18,3))
    axs = axs.flatten()

    axs[0].plot(disc_loss, color='blue')
    axs[0].set_title('Discriminator Loss')
    axs[0].set_ylabel('Loss')
    axs[0].set_xlabel('Model Update')

    axs[1].plot(disc_acc, color='lightblue')
    axs[1].axhline(0.5, color='red', linestyle='--', linewidth=1.5)
    axs[1].set_title('Discriminator Accuracy')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_xlabel('Model Update')
    axs[1].set_ylim(0,1.0)

    axs[2].plot(disc_acc_diff, color='blue')
    axs[2].set_title('Accuracy difference: First vs. Second half of Sequence')
    axs[2].set_ylabel('Accuracy Difference')
    axs[2].set_xlabel('Episode')
    
    plt.show()

def calculate_statistics(trajectories):
    means = []
    variances = []
    std_devs = []
    skewness = []
    kurtos = []

    for traj in trajectories:
        means.append(np.mean(traj))
        variances.append(np.var(traj))
        std_devs.append(np.std(traj))
        skewness.append(skew(traj))
        kurtos.append(kurtosis(traj))
    
    return means, variances, std_devs, skewness, kurtos

def plot_three_moments(supervised_means, real_means, bin_edges_mean,
                       supervised_variances, real_variances, bin_edges_var,
                       supervised_skew, real_skew, bin_edges_skew,
                       gail_means, gail_variances, gail_skew):
    fig, axs = plt.subplots(1, 3, figsize=(11, 3))

    # Plot 1: Mean Distribution
    axs[0].hist(supervised_means, bins=bin_edges_mean, alpha=0.8, label='Supervised', color='red')
    axs[0].hist(gail_means, bins=bin_edges_mean, alpha=0.8, label='GAIL', color='green')
    axs[0].hist(real_means, bins=bin_edges_mean, alpha=0.9, label='Real', color='skyblue')
    axs[0].legend(loc='upper right')
    axs[0].set_title('Mean')
    axs[0].set_xlabel('Value')
    axs[0].set_ylabel('Frequency')

    # Plot 1: Variance Distribution
    axs[1].hist(supervised_variances, bins=bin_edges_var, alpha=0.8, label='Supervised', color='red')
    axs[1].hist(gail_variances, bins=bin_edges_var, alpha=0.8, label='GAIL', color='green')
    axs[1].hist(real_variances, bins=bin_edges_var, alpha=0.9, label='Real', color='skyblue')
    axs[1].legend(loc='upper right')
    axs[1].set_title('Variance')
    axs[1].set_xlabel('Value')
    axs[1].tick_params(labelleft=False)

    # Plot 2: Skewness Distribution
    axs[2].hist(supervised_skew, bins=bin_edges_skew, alpha=0.8, label='Supervised', color='red')
    axs[2].hist(gail_skew, bins=bin_edges_skew, alpha=0.8, label='GAIL', color='green')
    axs[2].hist(real_skew, bins=bin_edges_skew, alpha=0.9, label='Real', color='skyblue')
    axs[2].legend(loc='upper right')
    axs[2].set_title('Skewness')
    axs[2].set_xlabel('Value')
    axs[2].tick_params(labelleft=False)
    plt.subplots_adjust(wspace=0.05)  # Reduce space between plots
    plt.tight_layout()
    plt.show()

def plot_sequences(real_seq, supervised, gail, token_size):
    # Create the subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))  # 1 row, 3 columns
    axes = axes.flatten()

    # Plot the real and simulated trajectories
    for i in range(3):
        axes[i].plot(real_seq[i][1:], label='Real Trajectory', color='skyblue')
        axes[i].plot(supervised[i], label='Supervised', color='red')
        axes[i].plot(gail[i], label='GAIL', color='green')
        axes[i].legend(loc='best', prop={'size': 10})
        axes[i].set_title(f'Trajectory {i+1}')
        axes[i].set_ylim(0-10, token_size+10)
        axes[i].set_xlabel('Time Step')
        
        if i == 0:  # Only set y-axis label on the first plot
            axes[i].set_ylabel('Price Delta')
        else:
            axes[i].tick_params(labelleft=False)  # Remove y-axis labels and ticks for the 2nd and 3rd plots

    # Adjust spacing between plots to bring them closer
    plt.subplots_adjust(wspace=0)  # Reduce space between plots
    plt.tight_layout()
    plt.show()

def wasserstein_dist(real_sequences, supervised, gail):

    n_timesteps = real_sequences.shape[1]
    
    # Calculate Wasserstein distance for each timestep
    wasserstein_supervised = []
    wasserstein_gail = []

    for t in range(n_timesteps):
        w_dist = wasserstein_distance(real_sequences[:, t], supervised[:, t])
        wasserstein_supervised.append(w_dist)
    
    for t in range(n_timesteps):
        w_dist = wasserstein_distance(real_sequences[:, t], gail[:, t])
        wasserstein_gail.append(w_dist)
    
    # Create visualization
    fig = plt.figure(figsize=(15, 3.5))
    
    # 1. Main grid for plots
    gs = plt.GridSpec(1, 3)
    
    # Plot 1: Wasserstein distances over time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(wasserstein_supervised, label = "Supervised")
    ax1.plot(wasserstein_gail, label = 'GAIL')
    ax1.legend()
    ax1.set_title('Wasserstein Distance by Timestep')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Wasserstein Distance')
    

    # Plot 2: Compare specific timesteps
    ax2 = fig.add_subplot(gs[0, 1])
    # Pick timestep with largest Wasserstein distance
    worst_timestep_supervised = np.argmax(wasserstein_supervised)

    # Get data for the specified timestep
    real_data = real_sequences[:, worst_timestep_supervised]
    generated_data = supervised[:, worst_timestep_supervised]

    # Calculate KDE
    kde_real = gaussian_kde(real_data)
    kde_generated = gaussian_kde(generated_data)

    # Create evaluation points
    x_eval = np.linspace(min(real_data.min(), generated_data.min()),
                        max(real_data.max(), generated_data.max()),
                        200)
    
    # Plot distributions
    ax2.plot(x_eval, kde_real(x_eval), label='Real', color='blue')
    ax2.plot(x_eval, kde_generated(x_eval), label='Supervised', color='red')
    ax2.set_title(f'Distribution Comparison at Timestep {worst_timestep_supervised}\n'
                f'(Largest Wasserstein Distance)')
    ax2.legend()

    # Plot 3: Compare specific timesteps
    ax3 = fig.add_subplot(gs[0, 2])
    # Pick timestep with largest Wasserstein distance
    worst_timestep_gail = np.argmax(wasserstein_gail)

    # Get data for the specified timestep
    real_data = real_sequences[:, worst_timestep_gail]
    generated_data = gail[:, worst_timestep_gail]

    # Calculate KDE
    kde_real = gaussian_kde(real_data)
    kde_generated = gaussian_kde(generated_data)

    # Create evaluation points
    x_eval = np.linspace(min(real_data.min(), generated_data.min()),
                        max(real_data.max(), generated_data.max()),
                        200)
    
    # Plot distributions
    ax3.plot(x_eval, kde_real(x_eval), label='Real', color='blue')
    ax3.plot(x_eval, kde_generated(x_eval), label='GAIL', color='green')
    ax3.set_title(f'Distribution Comparison at Timestep {worst_timestep_gail}\n'
                f'(Largest Wasserstein Distance)')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

def distribution_evolution(real_sequences, supervised, gail, token_size):
    n_timesteps = real_sequences.shape[1]
    bins = token_size

    # Calculate histogram for each timestep
    hist_data_real = []
    hist_data_supervised = []
    hist_data_gail = []
    
    # Define common range for all histograms
    global_min = min(real_sequences.min(), supervised.min(), gail.min())
    global_max = max(real_sequences.max(), supervised.max(), gail.max())
    
    for t in range(n_timesteps):
        hist_real, _ = np.histogram(real_sequences[:, t], bins=bins, 
                                  density=True, range=(global_min, global_max))
        hist_supervised, _ = np.histogram(supervised[:, t], bins=bins, 
                                        density=True, range=(global_min, global_max))
        hist_gail, _ = np.histogram(gail[:, t], bins=bins, 
                                  density=True, range=(global_min, global_max))
        
        hist_data_real.append(hist_real)
        hist_data_supervised.append(hist_supervised)
        hist_data_gail.append(hist_gail)

    # Convert to arrays
    hist_data_real = np.array(hist_data_real).T
    hist_data_supervised = np.array(hist_data_supervised).T
    hist_data_gail = np.array(hist_data_gail).T
    
    # Find global min and max for color scaling
    vmin = min(hist_data_real.min(), hist_data_supervised.min(), hist_data_gail.min())
    vmax = max(hist_data_real.max(), hist_data_supervised.max(), hist_data_gail.max())
    
    # Create visualization
    fig = plt.figure(figsize=(20, 3.5))
    gs = plt.GridSpec(1, 3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(hist_data_real, ax=ax1, cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_title('Real Sequences Distribution Evolution')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Value Distribution')

    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(hist_data_supervised, ax=ax2, cmap='viridis', vmin=vmin, vmax=vmax)
    ax2.set_title('Distribution Evolution Supervised')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Value Distribution')

    ax3 = fig.add_subplot(gs[0, 2])
    sns.heatmap(hist_data_gail, ax=ax3, cmap='viridis', vmin=vmin, vmax=vmax)
    ax3.set_title('Distribution Evolution GAIL')
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Value Distribution')
    
    plt.tight_layout()
    plt.show()

def distribution_evolution_strong_contrast(real_sequences, supervised, gail, token_size):
    n_timesteps = real_sequences.shape[1]
    bins = token_size

    # Calculate histograms (same as before)
    hist_data_real = []
    hist_data_supervised = []
    hist_data_gail = []
    
    global_min = min(real_sequences.min(), supervised.min(), gail.min())
    global_max = max(real_sequences.max(), supervised.max(), gail.max())
    
    for t in range(n_timesteps):
        hist_real, _ = np.histogram(real_sequences[:, t], bins=bins, 
                                  density=True, range=(global_min, global_max))
        hist_supervised, _ = np.histogram(supervised[:, t], bins=bins, 
                                        density=True, range=(global_min, global_max))
        hist_gail, _ = np.histogram(gail[:, t], bins=bins, 
                                  density=True, range=(global_min, global_max))
        
        hist_data_real.append(hist_real)
        hist_data_supervised.append(hist_supervised)
        hist_data_gail.append(hist_gail)

    # Convert to arrays
    hist_data_real = np.array(hist_data_real).T
    hist_data_supervised = np.array(hist_data_supervised).T
    hist_data_gail = np.array(hist_data_gail).T
    
    # Find global min and max
    vmin = min(hist_data_real.min(), hist_data_supervised.min(), hist_data_gail.min())
    vmax = max(hist_data_real.max(), hist_data_supervised.max(), hist_data_gail.max())
    
    # Create custom normalization to enhance color contrast
    # Option 1: Using power-law normalization
    from matplotlib.colors import PowerNorm
    norm = PowerNorm(gamma=0.5)  # gamma < 1 will enhance lower values
    
    # Option 2: Alternative - use LogNorm for even more contrast
    # from matplotlib.colors import LogNorm
    # norm = LogNorm(vmin=max(vmin, 0.001), vmax=vmax)  # avoid log(0)
    
    fig = plt.figure(figsize=(20, 3.5))
    gs = plt.GridSpec(1, 3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(hist_data_real, ax=ax1, cmap='viridis', 
                norm=norm, vmin=vmin, vmax=vmax)
    ax1.set_title('Real Sequences Distribution Evolution')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Value Distribution')

    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(hist_data_supervised, ax=ax2, cmap='viridis', 
                norm=norm, vmin=vmin, vmax=vmax)
    ax2.set_title('Distribution Evolution Supervised')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Value Distribution')

    ax3 = fig.add_subplot(gs[0, 2])
    sns.heatmap(hist_data_gail, ax=ax3, cmap='viridis', 
                norm=norm, vmin=vmin, vmax=vmax)
    ax3.set_title('Distribution Evolution GAIL')
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Value Distribution')
    
    plt.tight_layout()
    plt.show()

def token_distribution_distance(real_seq, generated_seq, n_tokens):
    """
    Calculate average Jensen-Shannon distance between token distributions at each timestep.
    """
    n_timesteps = real_seq.shape[1]
    distances = []
    
    for t in range(n_timesteps):
        # Calculate frequencies for this timestep
        freq1 = np.bincount(real_seq[:, t].astype(int), minlength=n_tokens) / len(real_seq)
        freq2 = np.bincount(generated_seq[:, t].astype(int), minlength=n_tokens) / len(generated_seq)
        
        # Add small epsilon to avoid log(0)
        freq1 += 1e-10
        freq2 += 1e-10
        
        # Normalize
        freq1 /= freq1.sum()
        freq2 /= freq2.sum()
        
        # Calculate Jensen-Shannon distance
        m = 0.5 * (freq1 + freq2)
        js_distance = 0.5 * (entropy(freq1, m) + entropy(freq2, m))
        distances.append(np.sqrt(js_distance))
    
    return np.mean(distances)

def transition_matrix_difference(real_seq, generated_seq, n_tokens):
    """
    Calculate average difference between transition matrices across timesteps.
    """
    def get_transition_matrices(sequence):
        n_timesteps = sequence.shape[1]
        transitions = []
        
        for t in range(n_timesteps - 1):
            current_tokens = sequence[:, t].astype(int)
            next_tokens = sequence[:, t + 1].astype(int)
            
            # Create transition matrix for this timestep
            trans_matrix = np.zeros((n_tokens, n_tokens))
            for curr, next_t in zip(current_tokens, next_tokens):
                trans_matrix[curr, next_t] += 1
            
            # Normalize
            row_sums = trans_matrix.sum(axis=1)
            row_sums[row_sums == 0] = 1
            trans_matrix = trans_matrix / row_sums[:, np.newaxis]
            transitions.append(trans_matrix)
            
        return transitions
    
    trans_real = get_transition_matrices(real_seq)
    trans_gen = get_transition_matrices(generated_seq)
    
    differences = [np.mean(np.abs(r - g)) for r, g in zip(trans_real, trans_gen)]
    return np.mean(differences)

def pattern_similarity_score(real_seq, generated_seq, window_size=5):
    """
    Calculate pattern similarity across the sequences.
    """
    def get_pattern_distribution(seq, window_size):
        n_samples = seq.shape[0]
        n_timesteps = seq.shape[1]
        patterns = []
        
        # Get patterns for each sample
        for i in range(n_samples):
            for t in range(n_timesteps - window_size + 1):
                pattern = tuple(seq[i, t:t+window_size].astype(int))
                patterns.append(pattern)
        
        unique, counts = np.unique(patterns, axis=0, return_counts=True)
        return dict(zip(map(tuple, unique), counts / len(patterns)))
    
    # Get distributions
    dist1 = get_pattern_distribution(real_seq, window_size)
    dist2 = get_pattern_distribution(generated_seq, window_size)
    
    # Get all unique patterns
    all_patterns = set(dist1.keys()) | set(dist2.keys())
    
    # Calculate Euclidean distance
    squared_diff_sum = 0
    for pattern in all_patterns:
        prob1 = dist1.get(pattern, 0)
        prob2 = dist2.get(pattern, 0)
        squared_diff_sum += (prob1 - prob2) ** 2
    
    return np.sqrt(squared_diff_sum)

def compare_sequences(real_data, supervised_data, gail_data, n_tokens):
    """
    Compare supervised and GAIL sequences against real data.
    """
    metrics = {
        'Method': ['Supervised', 'GAIL'],
        'Distribution Distance': [
            token_distribution_distance(real_data, supervised_data, n_tokens),
            token_distribution_distance(real_data, gail_data, n_tokens)
        ],
        'Transition Difference': [
            transition_matrix_difference(real_data, supervised_data, n_tokens),
            transition_matrix_difference(real_data, gail_data, n_tokens)
        ],
        'Pattern Similarity': [
            pattern_similarity_score(real_data, supervised_data),
            pattern_similarity_score(real_data, gail_data)
        ]
    }
    
    results_df = pd.DataFrame(metrics)
    
    # Print formatted results
    print("\nComparison Results:")
    print("==================")
    print(results_df.to_string(index=False))
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['Distribution Distance', 'Transition Difference', 'Pattern Similarity']
    
    bar_width = 0.35
    x = np.arange(len(metrics_to_plot))
    
    plt.bar(x - bar_width/2, results_df.iloc[0, 1:], bar_width, label='Supervised')
    plt.bar(x + bar_width/2, results_df.iloc[1, 1:], bar_width, label='GAIL')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score (lower is better)')
    plt.title('Comparison of Supervised vs GAIL Generation')
    plt.xticks(x, metrics_to_plot, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return results_df

def wasserstein_dist_new(real_sequences, supervised, gail):
    n_timesteps = real_sequences.shape[1]
    
    # Calculate Wasserstein distance for each timestep
    wasserstein_supervised = []
    wasserstein_gail = []

    for t in range(n_timesteps):
        # Calculate standard deviation of real data at this timestep
        real_var = np.std(real_sequences[:, t])
        
        # Calculate Wasserstein distance and normalize by standard deviation
        w_dist_supervised = wasserstein_distance(real_sequences[:, t], supervised[:, t])
        w_dist_gail = wasserstein_distance(real_sequences[:, t], gail[:, t])
        
        # Store normalized distances (as percentages)
        wasserstein_supervised.append((w_dist_supervised / real_var))
        wasserstein_gail.append((w_dist_gail / real_var))
    
    # Create visualization
    fig = plt.figure(figsize=(6, 3.5))
    gs = plt.GridSpec(1, 1)
    
    # Plot with updated y-axis label
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(wasserstein_supervised, label="Supervised")
    ax1.legend()
    ax1.set_title('Normalized Wasserstein Distance by Timestep')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Wasserstein Distance (% of SD)')
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=1))

def plot_n_distributions(real_data, supervised_data, gail_data, n_steps=10):
    # Create visualization
    plots_per_row = 5
    rows = (n_steps + plots_per_row - 1) // plots_per_row
    
    fig = plt.figure(figsize=(20, 4 * rows))
    gs = plt.GridSpec(rows, plots_per_row)
    
    for t in range(n_steps):
        # Calculate subplot position
        row = t // plots_per_row
        col = t % plots_per_row
        
        # Create subplot
        ax = fig.add_subplot(gs[row, col])
        
        # Get data for the timestep
        real_timestep = real_data[:, t]
        supervised_timestep = supervised_data[:, t]
        gail_timestep = gail_data[:, t]
        
        # Calculate KDE
        kde_real = gaussian_kde(real_timestep)
        kde_supervised = gaussian_kde(supervised_timestep)
        kde_gail = gaussian_kde(gail_timestep)
        
        # Create evaluation points
        x_eval = np.linspace(
            min(real_timestep.min(), supervised_timestep.min(), gail_timestep.min()),
            max(real_timestep.max(), supervised_timestep.max(), gail_timestep.max()),
            200
        )
        
        # Plot distributions
        ax.plot(x_eval, kde_real(x_eval), label='Real', color='blue', linewidth=0.8)
        ax.plot(x_eval, kde_supervised(x_eval), label='Supervised', color='red', linewidth=0.8)
        ax.plot(x_eval, kde_gail(x_eval), label='GAIL', color='green', linewidth=0.8)
        ax.set_title(f'Timestep {t}')
        ax.legend()
    
    plt.tight_layout()
    plt.show()

