Create a folder named random_field_dataset, include the following files:
## 1.1 generate_initial_condition.py;
Contain a function that takes args: (N_sample: int, Nx: int, domain: List, boundary_type:str, process_type:str, process_args**). Every call return a (N_sample, Nx) matrix that return N_sample samples generated from process_type using process_args and satisfy required boundary condition for Burgers equation;
boundary_type: support "periodic" (default) and "dirichlet"
Nx: 512 (default)
domain: [-1,1] (default)
process_type: support truncated_Gaussian, student_t, 
Note that you should create a single class for each type of process
truncated_Gaussian(trunc_dim, sigma, alpha, s):
Generate Gaussian process with reduced kernel to trunc_dim where eigenvalue $\lambda_j​=\sigma^2(\alpha^2+(j\pi)^2)^{−s}$.
student_t(nu=4, ...), the rest args are the same as truncated_Gaussian, i.e. use the same kernel as Gaussian process. 


## 1.2 burgers_solver.py, 
receive (initial_condition, visc, T), return solution at time T. The method Use Fourier spectral method + ETDRK4. The time step should be inferred automatically. Note that initial_condition should not only support pass 1 dimension vector, but also support batch solver, when pass shape like (N_sample, Nx).

## 1.3 generate_dataset.py,
First use functions from 1.1 to generate initial condition (N_process*N_sample, Nx), then call solver in 1.2 to generate solution at time T (use pytorch, when cuda is supported, use cuda to accelerate, else use cpu vectorization).

Save initial_condition and solution at time T as hdf5 format, named "initial" and "solution"

## 1.4 config 
Set config args from a json file that contains:
"initial_condition": args in 1.1 
"process_type": set default as truncated_Gaussian,
"process_args_range": Give sample range for process_args in 1.1; For truncated_Gaussian, set trunc_dim=8, sigma uniform [0.5,1.5], alpha loguniform[1,10], s uniform[1.5,4]; For student_t, use the same range.
"N_process":int, decide how much process need to sample from above range;
"solver": visc and T;

Make sure I can set all the important args through config file.

# Data generation
Call generate_dataset.py, sample 10 different process_args for each arg (sigma, alpha, s), total 1000 args, then
1. use these args to generate 1000 student_t process, set N_sample=200, Nx=256. Directly save 500 of each as initial.hdf5 file (don't use solver).
2. Use the rest 500 processes as initial condition, call solver for another 500 of each to generate the solution, save only the solution at time T as solution.hdf5.