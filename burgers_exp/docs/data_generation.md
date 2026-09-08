# Data generation
Call generate_dataset.py, sample different process_args for each arg (sigma, alpha, s), 14 sigma, 10 alpha, 10 s, totally 1400 args combination, 
1. use args combination to generate 1400 student_t process, set N_sample=400, Nx=256. Directly save 200 process samples (half of the 400 samples) as initial.hdf5 file (don't use solver).
2. Use the rest half process samples as initial condition, call solver to generate the solution, save only the solution at time T as solution.hdf5


 

