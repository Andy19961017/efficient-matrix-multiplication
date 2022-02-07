# Efficient Matrix Multiplication

In this project, implemented an efficient, single-threaded algorithm of matrix multiplication. Given matrices ```A```, ```B```, ```C``` of dimension ```lda × lda```, we want to implement the update ```C += A × B```. The three matrices are stored in column-major order. We tested the performance of our algorithm on Cori supercomputer at National Energy Research Scientific Computing Center. The performance of the algorithm is measured by the utilization of the CPU flop rate (i.e., the actual flop rate when performing the computation over the theoretical peak performance of the CPU). Naive implemenation only achieves ~1.0% of the peak performance while our implementation was able to achieve ~28.6%. The chart below shows the performance benchmark of our implementation versus naive implementation across different input matrix sizes.

<p align="center">
<img width="569" alt="benchmark" src="https://user-images.githubusercontent.com/37168711/152744868-750c7e8d-036f-4bef-a02e-b5878ffc32f2.png">
</p>
