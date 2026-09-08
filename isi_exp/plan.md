# Requirement:
1. Remove reliance on hydra and omegaconf package.
2. Keep the main generation logic in generation.py, but only 
save context_particles as input_dist, output_particles as output_dist, target_mean and target_cov, the rest logic and savings should be removed.
3. Use main.py to generate dataset, accept args: data_size (default 10), sample_size(default 5). Note that the correct generation would generate
input_dist: (data_size, sample_size, 4)
output_dist: (data_size, sample_size, 4)
target_mean: (data_size, 4)
target_cov: (data_size, 4, 4); all the data should be saved as hdf5 format.
4. The rest logic and unnecessary files should be removed.