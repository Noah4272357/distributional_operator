1. Modify the src/models/random_field_operator.py file:
1.1 Use a 5 layer neural network with input channel: pca_dim_in, and output channels feature=80, GeLU activation, map the shape into (N_random_field, N_sample, feature); 
1.2 Calculate the mean on N_sample dimension
1.3 Use a MLP to map feature to coeff_dim=n_Gaussian*(pca_dim_out\*(3+pca_dim_out)/2+1);
1.4 Using PCA from training dataset labels, recover shape into (N_random_field, coeff_dim , Nx);
1.5 Use output to construct 
\[\sum_{i=1}^{n} \pi_i\mathcal{GP}(m_i,K_i)\]



