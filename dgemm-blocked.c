#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <stdio.h>

#ifndef L_KERNEL_SIZE
#define L_KERNEL_SIZE 16
#endif

#ifndef M_BLOCK_SIZE
#define M_BLOCK_SIZE 2048
#endif
// Must be multiple of 8

#ifndef N_BLOCK_SIZE
#define N_BLOCK_SIZE 2048
#endif
// Must be multiple of L_KERNEL_SIZE

#ifndef K_BLOCK_SIZE
#define K_BLOCK_SIZE 2048
#endif
// Must be multiple of 8


#define min(a, b) (((a) < (b)) ? (a) : (b))

const char* dgemm_desc = "Blocked dgemm with SIMD matmul";


/*
L = 16
res = ""
res += f"void matmul_8_by_{L}_microkernel(\n"
res += "    double* A_topleft,\n"
res += "    double* B_topleft,\n"
res += "    int K,\n"
res += "    int A_nrows,\n"
res += "    int A_ncols,\n"
res += "    int B_nrows,\n"
res += "    int B_ncols,\n"
res += "    double* res\n"
res += "    ){\n"
res += "\n"
res += "	__m512d A"
for _ in range(L):
	res += f", acum_{_}"
for _ in range(L):
	res += f", B_{_}"
res += ";\n"
for _ in range(L):
	res += f"    acum_{_} = _mm512_load_pd(res + {_} * 8);\n"
res += "\n"
res += "    for(int k = 0; k < K; k += 1){\n"
res += "        #define PFETCH_DIST 12\n"
res += "        _mm_prefetch(A_topleft + (k + PFETCH_DIST) * 8, _MM_HINT_T0);\n"
res += "        _mm_prefetch(B_topleft + L_KERNEL_SIZE * (k + PFETCH_DIST) + 0, _MM_HINT_T0);\n"
res += "        _mm_prefetch(B_topleft + L_KERNEL_SIZE * (k + PFETCH_DIST) + 8, _MM_HINT_T0);\n"
res += "        A = _mm512_load_pd(A_topleft + k * 8);\n"
for _ in range(L):
	res += f"        B_{_} = _mm512_set1_pd(*(B_topleft + L_KERNEL_SIZE * k + {_}));\n"
for _ in range(L):
	res += f"        acum_{_} = _mm512_fmadd_pd(A, B_{_}, acum_{_});\n"
res += "    }\n"
for _ in range(L):
	res += f"    _mm512_store_pd(res + {_} * 8, acum_{_});\n"
res += "}\n"
print(res)
*/
void matmul_8_by_16_microkernel(
    double* A_topleft,
    double* B_topleft,
    int K,
    int A_nrows,
    int A_ncols,
    int B_nrows,
    int B_ncols,
    double* res,
    double* C_topleft,
    int lda,
    int save_to_C
    ){

	__m512d A, acum_0, acum_1, acum_2, acum_3, acum_4, acum_5, acum_6, acum_7, acum_8, acum_9, acum_10, acum_11, acum_12, acum_13, acum_14, acum_15, B_0, B_1, B_2, B_3, B_4, B_5, B_6, B_7, B_8, B_9, B_10, B_11, B_12, B_13, B_14, B_15;
    if(save_to_C){
        acum_0 =  _mm512_loadu_pd(C_topleft +  0 * lda);
        acum_1 =  _mm512_loadu_pd(C_topleft +  1 * lda);
        acum_2 =  _mm512_loadu_pd(C_topleft +  2 * lda);
        acum_3 =  _mm512_loadu_pd(C_topleft +  3 * lda);
        acum_4 =  _mm512_loadu_pd(C_topleft +  4 * lda);
        acum_5 =  _mm512_loadu_pd(C_topleft +  5 * lda);
        acum_6 =  _mm512_loadu_pd(C_topleft +  6 * lda);
        acum_7 =  _mm512_loadu_pd(C_topleft +  7 * lda);
        acum_8 =  _mm512_loadu_pd(C_topleft +  8 * lda);
        acum_9 =  _mm512_loadu_pd(C_topleft +  9 * lda);
        acum_10 = _mm512_loadu_pd(C_topleft + 10 * lda);
        acum_11 = _mm512_loadu_pd(C_topleft + 11 * lda);
        acum_12 = _mm512_loadu_pd(C_topleft + 12 * lda);
        acum_13 = _mm512_loadu_pd(C_topleft + 13 * lda);
        acum_14 = _mm512_loadu_pd(C_topleft + 14 * lda);
        acum_15 = _mm512_loadu_pd(C_topleft + 15 * lda);
    } else {
        acum_0 = _mm512_load_pd(res + 0 * 8);
        acum_1 = _mm512_load_pd(res + 1 * 8);
        acum_2 = _mm512_load_pd(res + 2 * 8);
        acum_3 = _mm512_load_pd(res + 3 * 8);
        acum_4 = _mm512_load_pd(res + 4 * 8);
        acum_5 = _mm512_load_pd(res + 5 * 8);
        acum_6 = _mm512_load_pd(res + 6 * 8);
        acum_7 = _mm512_load_pd(res + 7 * 8);
        acum_8 = _mm512_load_pd(res + 8 * 8);
        acum_9 = _mm512_load_pd(res + 9 * 8);
        acum_10 = _mm512_load_pd(res + 10 * 8);
        acum_11 = _mm512_load_pd(res + 11 * 8);
        acum_12 = _mm512_load_pd(res + 12 * 8);
        acum_13 = _mm512_load_pd(res + 13 * 8);
        acum_14 = _mm512_load_pd(res + 14 * 8);
        acum_15 = _mm512_load_pd(res + 15 * 8);
    }

    for(int k = 0; k < K; k += 1){
        #define PFETCH_DIST 12
        _mm_prefetch(A_topleft + (k + PFETCH_DIST) * 8, _MM_HINT_T0);
        _mm_prefetch(B_topleft + L_KERNEL_SIZE * (k + PFETCH_DIST) + 0, _MM_HINT_T0);
        _mm_prefetch(B_topleft + L_KERNEL_SIZE * (k + PFETCH_DIST) + 8, _MM_HINT_T0);
        A = _mm512_load_pd(A_topleft + k * 8);
        B_0 = _mm512_set1_pd(*(B_topleft + L_KERNEL_SIZE * k + 0));
        B_1 = _mm512_set1_pd(*(B_topleft + L_KERNEL_SIZE * k + 1));
        B_2 = _mm512_set1_pd(*(B_topleft + L_KERNEL_SIZE * k + 2));
        B_3 = _mm512_set1_pd(*(B_topleft + L_KERNEL_SIZE * k + 3));
        B_4 = _mm512_set1_pd(*(B_topleft + L_KERNEL_SIZE * k + 4));
        B_5 = _mm512_set1_pd(*(B_topleft + L_KERNEL_SIZE * k + 5));
        B_6 = _mm512_set1_pd(*(B_topleft + L_KERNEL_SIZE * k + 6));
        B_7 = _mm512_set1_pd(*(B_topleft + L_KERNEL_SIZE * k + 7));
        B_8 = _mm512_set1_pd(*(B_topleft + L_KERNEL_SIZE * k + 8));
        B_9 = _mm512_set1_pd(*(B_topleft + L_KERNEL_SIZE * k + 9));
        B_10 = _mm512_set1_pd(*(B_topleft + L_KERNEL_SIZE * k + 10));
        B_11 = _mm512_set1_pd(*(B_topleft + L_KERNEL_SIZE * k + 11));
        B_12 = _mm512_set1_pd(*(B_topleft + L_KERNEL_SIZE * k + 12));
        B_13 = _mm512_set1_pd(*(B_topleft + L_KERNEL_SIZE * k + 13));
        B_14 = _mm512_set1_pd(*(B_topleft + L_KERNEL_SIZE * k + 14));
        B_15 = _mm512_set1_pd(*(B_topleft + L_KERNEL_SIZE * k + 15));
        acum_0 = _mm512_fmadd_pd(A, B_0, acum_0);
        acum_1 = _mm512_fmadd_pd(A, B_1, acum_1);
        acum_2 = _mm512_fmadd_pd(A, B_2, acum_2);
        acum_3 = _mm512_fmadd_pd(A, B_3, acum_3);
        acum_4 = _mm512_fmadd_pd(A, B_4, acum_4);
        acum_5 = _mm512_fmadd_pd(A, B_5, acum_5);
        acum_6 = _mm512_fmadd_pd(A, B_6, acum_6);
        acum_7 = _mm512_fmadd_pd(A, B_7, acum_7);
        acum_8 = _mm512_fmadd_pd(A, B_8, acum_8);
        acum_9 = _mm512_fmadd_pd(A, B_9, acum_9);
        acum_10 = _mm512_fmadd_pd(A, B_10, acum_10);
        acum_11 = _mm512_fmadd_pd(A, B_11, acum_11);
        acum_12 = _mm512_fmadd_pd(A, B_12, acum_12);
        acum_13 = _mm512_fmadd_pd(A, B_13, acum_13);
        acum_14 = _mm512_fmadd_pd(A, B_14, acum_14);
        acum_15 = _mm512_fmadd_pd(A, B_15, acum_15);
    }
    if(save_to_C){
        _mm512_storeu_pd(C_topleft +  0 * lda, acum_0);
        _mm512_storeu_pd(C_topleft  + 1 * lda, acum_1);
        _mm512_storeu_pd(C_topleft  + 2 * lda, acum_2);
        _mm512_storeu_pd(C_topleft  + 3 * lda, acum_3);
        _mm512_storeu_pd(C_topleft  + 4 * lda, acum_4);
        _mm512_storeu_pd(C_topleft  + 5 * lda, acum_5);
        _mm512_storeu_pd(C_topleft  + 6 * lda, acum_6);
        _mm512_storeu_pd(C_topleft  + 7 * lda, acum_7);
        _mm512_storeu_pd(C_topleft  + 8 * lda, acum_8);
        _mm512_storeu_pd(C_topleft  + 9 * lda, acum_9);
        _mm512_storeu_pd(C_topleft + 10 * lda, acum_10);
        _mm512_storeu_pd(C_topleft + 11 * lda, acum_11);
        _mm512_storeu_pd(C_topleft + 12 * lda, acum_12);
        _mm512_storeu_pd(C_topleft + 13 * lda, acum_13);
        _mm512_storeu_pd(C_topleft + 14 * lda, acum_14);
        _mm512_storeu_pd(C_topleft + 15 * lda, acum_15);
    } else {
        _mm512_store_pd(res  + 0 * 8, acum_0);
        _mm512_store_pd(res  + 1 * 8, acum_1);
        _mm512_store_pd(res  + 2 * 8, acum_2);
        _mm512_store_pd(res  + 3 * 8, acum_3);
        _mm512_store_pd(res  + 4 * 8, acum_4);
        _mm512_store_pd(res  + 5 * 8, acum_5);
        _mm512_store_pd(res  + 6 * 8, acum_6);
        _mm512_store_pd(res  + 7 * 8, acum_7);
        _mm512_store_pd(res  + 8 * 8, acum_8);
        _mm512_store_pd(res  + 9 * 8, acum_9);
        _mm512_store_pd(res + 10 * 8, acum_10);
        _mm512_store_pd(res + 11 * 8, acum_11);
        _mm512_store_pd(res + 12 * 8, acum_12);
        _mm512_store_pd(res + 13 * 8, acum_13);
        _mm512_store_pd(res + 14 * 8, acum_14);
        _mm512_store_pd(res + 15 * 8, acum_15);
    }
}

/* Code to autogenerate "do_block". L is the parameter which should match to L_KERNEL_SIZE
L = 16
res = ""
res += "static void do_block(int lda, int A_nrows, int A_ncols, int B_nrows, int B_ncols, int M, int M_unpadded, int N, int N_unpadded, int K, double* A, double* B, double* C){\n"
res += "    double* res = (double*) _mm_malloc(8 * L_KERNEL_SIZE * sizeof(double), 64);\n"
res += "    for (int j = 0, B_shift = 0; j < N; j += L_KERNEL_SIZE, B_shift += L_KERNEL_SIZE * K){\n"
res += "        for (int i = 0, A_shift = 0; i < M; i += 8, A_shift += 8 * K){\n"
for _ in range(L):
    res += f"            memcpy(res + {_} * 8, C + i + (j + {_}) * lda, 8 * sizeof(double));\n"
res += f"            matmul_8_by_{L}_microkernel(\n"
res += "            A + A_shift,\n"
res += "            B + B_shift,\n"
res += "            K,\n"
res += "            A_nrows,\n"
res += "            A_ncols,\n"
res += "            B_nrows,\n"
res += "            B_ncols,\n"
res += "            res\n"
res += "            );\n"
res += "\n"
res += "            // Put data from res to C\n"
res += "            // Usually, the entire kernal outout has data\n"
res += "            // but, sometimes, only (M_unpadded - i) * (N_unpadded - j) has actual data\n"
res += "            int true_data_col_len = min(8, M_unpadded - i);\n"
res += "            int true_data_row_len = min(L_KERNEL_SIZE, N_unpadded - j);\n"
res += "            switch (true_data_row_len){\n"
for _ in range(L-1, -1, -1):
    res += f"                case {_+1}:\n"
    res += f"                    memcpy(C + i + (j + {_}) * lda, res + {_} * 8, true_data_col_len * sizeof(double));\n"
res += "            }\n"
res += "        }\n"
res += "    }\n"
res += "}\n"
print(res)
*/

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M x N, A is M x K, and B is K x N.
 * However, only M_unpadded x N_unpadded of C, M_unpadded x K of A, and K x N_unpadded has non-zero data
 */
static void do_block(int lda, int A_nrows, int A_ncols, int B_nrows, int B_ncols, int M, int M_unpadded, int N, int N_unpadded, int K, double* A, double* B, double* C){
    double* res = (double*) _mm_malloc(8 * L_KERNEL_SIZE * sizeof(double), 64);
    int save_to_C = 1;
    double* C_topleft;
    for (int j = 0, B_shift = 0; j < N; j += L_KERNEL_SIZE, B_shift += L_KERNEL_SIZE * K){
        for (int i = 0, A_shift = 0; i < M; i += 8, A_shift += 8 * K){
            // Determine if put data from res to C
            // Usually, the entire kernal outout has data
            // but, sometimes, only (M_unpadded - i) * (N_unpadded - j) has actual data
            int true_data_col_len = min(8, M_unpadded - i);
            int true_data_row_len = min(L_KERNEL_SIZE, N_unpadded - j);
            save_to_C = (true_data_col_len == 8) & (true_data_row_len == L_KERNEL_SIZE);
            C_topleft = C + i + (j + 0) * lda;

            if(!save_to_C){
                switch (true_data_row_len){
                    case 16:
                        memcpy(res + 15 * 8, C_topleft + 15 * lda, true_data_col_len * sizeof(double));
                    case 15:
                        memcpy(res + 14 * 8, C_topleft + 14 * lda, true_data_col_len * sizeof(double));
                    case 14:
                        memcpy(res + 13 * 8, C_topleft + 13 * lda, true_data_col_len * sizeof(double));
                    case 13:
                        memcpy(res + 12 * 8, C_topleft + 12 * lda, true_data_col_len * sizeof(double));
                    case 12:
                        memcpy(res + 11 * 8, C_topleft + 11 * lda, true_data_col_len * sizeof(double));
                    case 11:
                        memcpy(res + 10 * 8, C_topleft + 10 * lda, true_data_col_len * sizeof(double));
                    case 10:
                        memcpy(res + 9 * 8, C_topleft + 9 * lda, true_data_col_len * sizeof(double));
                    case 9:
                        memcpy(res + 8 * 8, C_topleft + 8 * lda, true_data_col_len * sizeof(double));
                    case 8:
                        memcpy(res + 7 * 8, C_topleft + 7 * lda, true_data_col_len * sizeof(double));
                    case 7:
                        memcpy(res + 6 * 8, C_topleft + 6 * lda, true_data_col_len * sizeof(double));
                    case 6:
                        memcpy(res + 5 * 8, C_topleft + 5 * lda, true_data_col_len * sizeof(double));
                    case 5:
                        memcpy(res + 4 * 8, C_topleft + 4 * lda, true_data_col_len * sizeof(double));
                    case 4:
                        memcpy(res + 3 * 8, C_topleft + 3 * lda, true_data_col_len * sizeof(double));
                    case 3:
                        memcpy(res + 2 * 8, C_topleft + 2 * lda, true_data_col_len * sizeof(double));
                    case 2:
                        memcpy(res + 1 * 8, C_topleft + 1 * lda, true_data_col_len * sizeof(double));
                    case 1:
                        memcpy(res + 0 * 8, C_topleft + 0 * lda, true_data_col_len * sizeof(double));
                }
            }

            matmul_8_by_16_microkernel(
                A + A_shift,
                B + B_shift,
                K,
                A_nrows,
                A_ncols,
                B_nrows,
                B_ncols,
                res,
                C + i + j * lda,
                lda,
                save_to_C
            );

            if(!save_to_C){
                switch (true_data_row_len){
                    case 16:
                        memcpy(C_topleft + 15 * lda, res + 15 * 8, true_data_col_len * sizeof(double));
                    case 15:
                        memcpy(C_topleft + 14 * lda, res + 14 * 8, true_data_col_len * sizeof(double));
                    case 14:
                        memcpy(C_topleft + 13 * lda, res + 13 * 8, true_data_col_len * sizeof(double));
                    case 13:
                        memcpy(C_topleft + 12 * lda, res + 12 * 8, true_data_col_len * sizeof(double));
                    case 12:
                        memcpy(C_topleft + 11 * lda, res + 11 * 8, true_data_col_len * sizeof(double));
                    case 11:
                        memcpy(C_topleft + 10 * lda, res + 10 * 8, true_data_col_len * sizeof(double));
                    case 10:
                        memcpy(C_topleft + 9 * lda, res + 9 * 8, true_data_col_len * sizeof(double));
                    case 9:
                        memcpy(C_topleft + 8 * lda, res + 8 * 8, true_data_col_len * sizeof(double));
                    case 8:
                        memcpy(C_topleft + 7 * lda, res + 7 * 8, true_data_col_len * sizeof(double));
                    case 7:
                        memcpy(C_topleft + 6 * lda, res + 6 * 8, true_data_col_len * sizeof(double));
                    case 6:
                        memcpy(C_topleft + 5 * lda, res + 5 * 8, true_data_col_len * sizeof(double));
                    case 5:
                        memcpy(C_topleft + 4 * lda, res + 4 * 8, true_data_col_len * sizeof(double));
                    case 4:
                        memcpy(C_topleft + 3 * lda, res + 3 * 8, true_data_col_len * sizeof(double));
                    case 3:
                        memcpy(C_topleft + 2 * lda, res + 2 * 8, true_data_col_len * sizeof(double));
                    case 2:
                        memcpy(C_topleft + 1 * lda, res + 1 * 8, true_data_col_len * sizeof(double));
                    case 1:
                        memcpy(C_topleft + 0 * lda, res + 0 * 8, true_data_col_len * sizeof(double));
                }
            }
        }
    }
    _mm_free(res);
}


/* 
 * This function take a matrix original_mtx of size lda x lda, and writes to
 * padded_mtx where zeros are padded to the bottom and the right such that
 * the padded_mtx has dimension nrows x ncols
 */
void pad_with_zeros(double* original_mtx, int lda, int nrows, int ncols, double* padded_mtx){
    memset(padded_mtx, 0, nrows * ncols * sizeof(double));
    for(int c = 0; c < lda; c++){
        memcpy(padded_mtx + c * nrows, original_mtx + c * lda, lda * sizeof(double));
    }
}

/* Repack the K x B_ncols block of B into B_tilde, zig-zagging in slices of size K * L_KERNEL_SIZE
 * Note that B_ncols is multiple of L_KERNEL_SIZE because of our repacking.
 */
void repack_B(int K, double* B_padded_topleft, int B_nrows, int B_ncols, double* B_tilde){
    int B_slices = (int)((B_ncols + L_KERNEL_SIZE - 1) / L_KERNEL_SIZE);
    for(int slice = 0; slice < B_slices; slice++){
        for(int j = 0; j < L_KERNEL_SIZE; j++){
            for(int i = 0; i < K; i++){
                B_tilde[(j + i * L_KERNEL_SIZE) + slice * K * L_KERNEL_SIZE] = B_padded_topleft[(i + j * B_nrows) + slice * B_nrows * L_KERNEL_SIZE];
            }
        }
    }
}

/* Repack the M x K block of A into A_tilde, zig-zagging in slices of size
 * 8 x K. Note that M is multiple of 8 because of our repacking.
 */
void repack_A(int K, int M, double* A_padded_topleft, int A_nrows, int A_ncols, double* A_tilde){
    int A_slices = (int)((M + 8 - 1) / 8); // A_slices = M / 8
    for(int slice = 0; slice < A_slices; slice++){
        for(int j = 0; j < K; j++){
            for(int i = 0; i < 8; i++){
                A_tilde[(i + j * 8) + (slice * 8 * K)] = A_padded_topleft[(i + j * A_nrows) + (slice * 8)];
            }
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C) {
    // First we create padded versions of A and B so SIMD won't cross boundaries.
    int A_nrows = 8 * (int)((lda + 8 - 1) / 8);
    int A_ncols = 8 * (int)((lda + 8 - 1) / 8);
    int B_nrows = 8 * (int)((lda + 8 - 1) / 8);
    int B_ncols = L_KERNEL_SIZE * (int)((lda + L_KERNEL_SIZE - 1) / L_KERNEL_SIZE);
    
    // This way A_padded has dimension A_nrows x A_ncols,
    // B_padded has dimension B_nrows x B_ncols,
    double* A_padded = (double*) _mm_malloc(A_nrows * A_ncols * sizeof(double), 64);
    double* B_padded = (double*) _mm_malloc(B_nrows * B_ncols * sizeof(double), 64);
    double* A_tilde = (double*) _mm_malloc(min(M_BLOCK_SIZE, A_nrows) * min(K_BLOCK_SIZE, A_ncols) * sizeof(double), 64);
    double* B_tilde = (double*) _mm_malloc(min(K_BLOCK_SIZE, A_ncols) * B_ncols * sizeof(double), 64);

    pad_with_zeros(A, lda, A_nrows, A_ncols, A_padded);
    pad_with_zeros(B, lda, B_nrows, B_ncols, B_padded);

    int A_n_block_rows = (int)((A_nrows + M_BLOCK_SIZE - 1) / M_BLOCK_SIZE);
    int A_n_block_cols = (int)((A_ncols + K_BLOCK_SIZE - 1) / K_BLOCK_SIZE);
    // int B_n_block_rows = (int)((B_nrows + K_BLOCK_SIZE - 1) / K_BLOCK_SIZE); // == A_n_block_cols
    int B_n_block_cols = (int)((B_ncols + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE); 

    for (int k = 0; k < A_n_block_cols; k += 1) {
        // Repack B_padded into B tilde
        int K = min(K_BLOCK_SIZE, A_ncols - k * K_BLOCK_SIZE);
        repack_B(K, B_padded + k * K_BLOCK_SIZE, B_nrows, B_ncols, B_tilde);
        for (int i = 0; i < A_n_block_rows; i += 1) {
            int M = min(M_BLOCK_SIZE, A_nrows - i * M_BLOCK_SIZE);
            repack_A(K, M, A_padded + k * K_BLOCK_SIZE * A_nrows + i * M_BLOCK_SIZE, A_nrows, A_ncols, A_tilde);
            // TODO: Multiply A_tilde by B_tilde. Will be a for(i) for(j) loop. Ignore N_BLOCK_SIZE (in fact, remove it)
            for (int j = 0; j < B_n_block_cols; j += 1) {
                // For each block-column of B
                // Accumulate block dgemms into block of C
                // Correct block dimensions if block "goes off edge of" the matrix
                // do_block() computes matrix multiplication between blocks of M x K and K x N
                // and gets a M x N result
                // but only M_unpadded x N_unpadded of the result has non-zero data
                int M_unpadded = min(M_BLOCK_SIZE, lda - i * M_BLOCK_SIZE);
                int N = min(N_BLOCK_SIZE, B_ncols - j * N_BLOCK_SIZE);
                int N_unpadded = min(N_BLOCK_SIZE, lda - j * N_BLOCK_SIZE);
                // Perform individual block dgemm
                do_block(lda, A_nrows, A_ncols, B_nrows, B_ncols, M, M_unpadded, N, N_unpadded, K, A_tilde, B_tilde + j * K * N_BLOCK_SIZE, C + i * M_BLOCK_SIZE + j * N_BLOCK_SIZE * lda);
            }
        }
    }
    _mm_free(A_padded);
    _mm_free(B_padded);
    _mm_free(A_tilde);
    _mm_free(B_tilde);
}