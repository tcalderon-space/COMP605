[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/TsA0lDHW)
# 1. SDSU Comp/CS 605 Spring 25 Assignment 5

In this assignment you will implement your own algorithm to reverse vectors using CUDA programming in Julia.

In [lecture 28](https://sdsu-comp605.github.io/spring25/lectures/module8-1_cuda_jl.html) in class, we learned how to set up and use the Julia CUDA interface, CUDA.jl.

In this assignment, you will use parallel GPU programming in Julia to reverse a vector $A$ in place:

$$
A_i := A_{N - i + 1},
$$

where $N$ is the length of the vector.

In the starter code, [reverse_vec.jl](reverse_vec.jl), we have four implementations:


1. A reference (CPU) for loop-based implementation as a baseline: `reversevec!`
2. A "fake" GPU kernel, where we manually loop over blocks of indices: `fake_knl_reversevecs!`
3. A real double-array reverse GPU kernel that uses the built-in CUDA variables for thread index, block index and block dimension: `knl_reversevecs!`
4. A real inplace reverse GPU kernel that uses the built-in CUDA variables for thread index, block index and block dimension: `knl_reversevecs_inplace_bad!`
- **Note:** The `knl_reversevecs_inplace_bad!` version has a problem!

In the testing code, we also synchronize the results between GPU and CPU and time the executions.

**Bandwidth analysis**:

The `knl_reversevecs_inplace_bad!` kernel will need to load all the data once and write all the data once, so the total data movement is $2 N$. Thus, memory bandwidth can be calculated as:

```math
\textrm{bandwidth} = 2 N * {\textrm{sizeof}(T)} / \textrm{time}
```
where ${\tt \textrm{time}}$ is the kernel runtime in seconds, and $\textrm{T}$ is the float type you are executing with (it could be `Float32`, `Float64`, etc.). This will give you bandwidth in bytes per second, you likely want to divide this by $1024^3$ to convert to Gigabytes per second (Gib/s).


## Assignment steps:

1. (25%) Write your own correct `knl_reversevecs_inplace!` kernel
2. (15%) Add testing and timing of this kernel. Your testing should include at least a correctness check.
3. (10%) In [`Report.ipynb`](Report.ipynb) explain in your own words what problem the `knl_reversevecs_inplace_bad!` version has.
4. (10%) In [`Report.ipynb`](Report.ipynb) provide a bandwidth analysis of the correct `knl_reversevecs_inplace!` kernel
5. (40%) In [`Report.ipynb`](Report.ipynb) add a performance analysis that should produce at least the following two figures and related commentary. Figures that should be included:
  5.1) Bandwidth vs problem size $N$ (you need to test for different values of $N$) for `Float64` and `Float32`.
  5.2) A roofline plot (reference [Lecture 6](https://sdsu-comp605.github.io/spring25/lectures/module2-1_measuring_performance.html)).


## Requirements:

- Your implementation should be written in Julia.
- Kernel computations should be performed on the GPU (if you do not have an NVIDIA GPU use the `tuckoo` cluster) using the `CUDA.jl` package.
- Your code should include checks/tests (this can either be done in the testing/driver part of the file itself or a separate testing file); this will likely be a comparison between the output of the CPU and GPU code.

### A non-exhaustive list of things that might get you points off

- Forgetting to open your PR.

- Not submitting all required files (see below the "Reminders on workflow, best practices and submission requirements" section).

- To avoid error-prone copying/pasting of your results to analyze your performance (see Part 5), you should consider modifying the testing code to print values to file rather than standart output (terminal) and then read data from file in your post-processing/data analysis code in [`Report.ipynb`](Report.ipynb).


## Reminders on workflow, best practices and submission requirements

- Only changes made within the deadline (including the lateness window) will be graded.

- There is no need to tag your instructor/TA as a Reviewer in your open PR. We will review your work when ready.

- Remember not to attempt to close your PR. It needs to stay open for Reviewers (in this case your instructor and TAs) to review and grade your work.

- Always remember to double check the `File changed` tab in your PR. If you see files that should not belong there (e.g., files automatically created by your IDE or virtual environment files) remove them. Also, it is the student's responsibility to make sure that code is working (i.e., does not crash) and everything you pushed/submitted looks exactly how you intended it to be (double check typos, formatting errors, etc.).

- If you are using an IDE that automatically creates hidden project files that you might inadvertently push to your branch, it is always a good practice to use a `.gitignore` file that specify which files you do _not_ want to be tracked by `git`, and therefore, pushed to your branch. Recall that we covered this in our [first lecture](https://sdsu-comp605.github.io/spring25/lectures/module1-1_first_class.html#git).

- You are required to submit not only your Julia code (i.e., the file with the `.jl` extension), but also the `Project.toml` and `Manifest.toml` files that are created when you instantiate your environment wehn you invoke `julia` with the `--project=.` option.

- For this assignment, if you are using the `tuckoo` cluster, you are also required to submit any SLURM batch job script that you used.

