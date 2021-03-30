**Better Bin Packing through Matrix Multiplication**

With these notebooks, I try a new approach to the bin-packing problem that takes advantage of matrix multiplication to try many different combinations of articles at the same time. In the classic one-dimensional bin problem, the goal is to load as many articles of varying length into bins that all have a standard length.
https://en.wikipedia.org/wiki/Bin_packing_problem 
There are many similar real-world problems related to literal packing and scheduling, and also related computer science problems. Many heuristic algorithms can approximate the optimal solution. My goal is to combine the best of both worlds: a solution that is fast and inexpensive while also optimal (or nearly so).

These notebooks use a fundamentally different approach, essentially using memory as a substitute for computation to try millions of combination at once. A matrix of booleans represents the different combinations and a vector contains the loads. By multiplying them together, we can get all the different combinations at once. The first versions use numpy on a CPU, but the next version will use PyTorch on GPUs so that the computer can try billions rather than millions of combinations.
