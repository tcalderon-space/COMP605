[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/B1K8JJS_)
# 1. SDSU Comp/CS 605 Spring 25 Assignment 4

In this assignment you will implement your own reduction algorithm using MPI in Julia.

In particular, you should write a code to compute the (vector / pointwise) sum of data in an array to a single value on the `root` MPI rank.


> Useful references:
> 1.  Article: Chan et al., [_Collective communication: theory, practice, and experience_](https://csu-sdsu.primo.exlibrisgroup.com/permalink/01CALS_SDL/10r4g1c/cdi_crossref_primary_10_1002_cpe_1206), Figure 3 (b)
> 2. Lecture Notes from the University of Texas at Austin: Robert van de Geijn (RVDG) [_Collective Communication: Theory and Practice_](https://www.cs.utexas.edu/~rvdg/tmp/CollectiveCommunication.pdf), pages 172-184


I have given you a naive implementation (in `naivereduce.jl`) and test (in `naivereduce_test.jl`), and you should write one that uses a Minimum Spanning Tree (MST) algorithm like what we did in [class for the broadcast](https://sdsu-comp605.github.io/spring25/lectures/module6-3_collectives.html).

If you're using your own machine to test this code with, assuming your machine has at least 4 cores, test your code under 4 MPI ranks with the following command line prompt:

```
mpiexec -n 4 julia --project=. your_test.jl
```

## Assignment steps:

1. (30%) Write your own MST reduce(to-one) algorithm to sum the rank IDs of all the ranks in the whole communicator. This version should use recursion (as we've seen in class for the MST broadcast)
2. (20%) Extend the testing code `naivereduce_test.jl` to compare the execution times of the naive implementation and your MST implementation (similar to what we did in class to compare the naive and MST broadcasts).
3. (15%) Use the tuckoo cluster to test the execution up to 16 MPI ranks. Use the SLURM batch job script provided in [`batch_scripts/batch.jello`](https://github.com/sdsu-comp605/spring25/blob/main/batch_scripts/batch.jello) as a template and modify it accordingly to launch your program.
4. (35%) In the attached `Report.ipynb` write a commentary and implement your own post-processing/data analysis of your execution results, producing 1 or 2 figures with detailed captions presenting the results of your study.
  - Do not include anything else in your report other than the commentary and figures!
  - Possible figure: Comparison of runtime versus problem size (number of MPI ranks) for the naive and Minimum Spanning Tree algorithms.


### A non-exhaustive list of things that might get you points off

- Not submitting all required files (see below the "Reminders on workflow, best practices and submission requirements" section).

- To avoid error-prone copying/pasting of your results to analyze your performance (see Part 4), you should consider modifying the testing code to print values to file rather than standart output (terminal) and then read data from file in your post-processing/data analysis code in `Report.ipynb`.

## Extra Credit:

1. (+5%) What are the total parallel costs (in terms of number of steps and cost per step) of the naive and MST reduce algorithms? For these, you want to use the same notation as in the references, i.e., $\alpha$ and $\beta$, respectively, represent the message startup time and per data item transmission time, $\gamma$ denotes the cost required to perform an arithmetic operation (e.g. a reduction operation), and $n$ is the length of the message.
2. (+20%) Develop your own _non-recursive_ MST reduce(to-one) algorithm and compare its execution results with the recursive MST version. Add a figure to your report with this comparison and comment on the observed performance.


## Reminders on workflow, best practices and submission requirements

- Only changes made within the deadline (including the lateness window) will be graded.

- There is no need to tag your instructor/TA as a Reviewer in your open PR. We will review your work when ready.

- Remember not to attempt to close your PR. It needs to stay open for Reviewers (in this case your instructor and TAs) to review and grade your work.

- Always remember to double check the `File changed` tab in your PR. If you see files that should not belong there (e.g., files automatically created by your IDE or virtual environment files) remove them. Also, it is the student's responsibility to make sure that code is working (i.e., does not crash) and everything you pushed/submitted looks exactly how you intended it to be (double check typos, formatting errors, etc.).

- If you are using an IDE that automatically creates hidden project files that you might inadvertently push to your branch, it is always a good practice to use a `.gitignore` file that specify which files you do _not_ want to be tracked by `git`, and therefore, pushed to your branch. Recall that we covered this in our [first lecture](https://sdsu-comp605.github.io/spring25/lectures/module1-1_first_class.html#git).

- You are required to submit not only your Julia code (i.e., the file with the `.jl` extension), but also the `Project.toml` and `Manifest.toml` files that are created when you instantiate your environment wehn you invoke `julia` with the `--project=.` option.

- For this assignment, you are also required to submit your SLURM batch job script.
