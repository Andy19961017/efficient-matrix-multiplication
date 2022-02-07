# Efficient Matrix Multiplication

## Introduction
In this project, implemented an efficient, single-threaded algorithm of matrix multiplication. It is implemented in C programming language. Given matrices ```A```, ```B```, ```C``` of dimension ```lda × lda```, we want to implement the update ```C += A × B```. The three matrices are stored in column-major order. We tested the performance of our algorithm on Cori supercomputer at National Energy Research Scientific Computing Center. The performance of the algorithm is measured by the utilization of the CPU flop rate (i.e., the actual flop rate when performing the computation over the theoretical peak performance of the CPU). Naive implemenation only achieves ~1.0% of the peak performance while our implementation was able to achieve ~28.6%. The chart below shows the performance benchmark of our implementation versus naive implementation across different input matrix sizes.

<p align="center">
<img width="569" alt="benchmark" src="https://user-images.githubusercontent.com/37168711/152744868-750c7e8d-036f-4bef-a02e-b5878ffc32f2.png">
</p>

## Technical Details
The key to achieve efficient matrix multiplication is to reduce the number memory read/write operations as memory is the bottle neck of the program (rather than CPU flop rate). This principle is applied to different levels of memory (as the figure below). Intuitively, when datas are loaded from RAM into cache or register, we want to perform as much computation as possible using the datas before releasing them from cache or register.
![maxresdefault](https://user-images.githubusercontent.com/37168711/152750417-d5240b1d-b73f-482c-aefc-4104c8842a64.jpeg)
The following techniques are used in our implementation:
### 1. Blocking
Matrix ```C``` is divided into blocks of ```M_BLOCK_SIZE × N_BLOCK_SIZE```, ```A``` is divided into blocks of ```M_BLOCK_SIZE × K_BLOCK_SIZE```, and ```B``` is divided into blocks of ```K_BLOCK_SIZE × N_BLOCK_SIZE```. As illustrated in the figure below, each block of ```C``` is derived from the product of a row of blocks of ```A``` and a column of blocks of ```B```. Blocking allows the program to be more **cache friendly**. During the computation, the blocks are loaded into the cache and will not be released before the entire block of ```C``` is calculated.
<p align="center">
<img width="662" alt="Screen Shot 2022-02-06 at 11 50 06 PM" src="https://user-images.githubusercontent.com/37168711/152746649-8492055e-5846-4ef4-af53-a6a3ffb4c727.png">
</p>
### 2. SIMD microkernel
We implemented a SIMD (Single Instruction Multiple Data) microkernel with 512-bit Intel Advanced Vector eXtensions (AVX512) intrinsics. This allows multiple datas being calculated in parallel as well as using the registers more efficiently. When the microkernel is executed, it loads a slice of ```A``` and ```B``` into the registers and computes the outer product to derive the associated block of ```C``` (illustrated in the figure below).
![outer_product](https://user-images.githubusercontent.com/37168711/152752360-8bb35ddb-18a1-4efa-bc2b-55d562368c9c.jpeg)
