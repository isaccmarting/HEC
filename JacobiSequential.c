#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 如果内存不足，返回值是ERROR
#define ERROR -1
#define PI 3.1415926
// 偏微分方程结果的输出文件
#define OUTPUT_FILENAME "result.txt"

// 矩阵结构
typedef struct MatrixConfigure {
    int width;         // 矩阵的列数
    int height;        // 矩阵的行数
    double** values;   // 矩阵的值
} *Matrix; 

// 初始化矩阵
void initialMatrix(Matrix mat); 
// 释放矩阵的内存空间
void freeMatrix(Matrix mat); 
// 求解偏微分方程的函数，这个是串行方法
void JacobiSequential(Matrix mat); 
// 计算结果与真实值之间的L2-norm
double costL2norm(Matrix mat); 
// 输出偏微分方程结果到指定文件
int outputValues(int height, int width, double** values); 

// N是网格的大小；numIters是迭代的次数
int N, numIters; 
// 迭代收敛的阈值（设置的值小，保证有numIters次迭代）
double eps; 
// 网格的步长
double step_x, step_y; 

int main(int argc, char* argv[])
{
    Matrix mat; 
    
    // 初始化设置
    N = 2500; numIters = 200; eps = 0.001; 
    step_x = step_y = 1.0 / (N-1); 
    // 初始化网格矩阵mat
    mat = (Matrix) malloc(sizeof(struct MatrixConfigure)); 
    if(mat == NULL){
        printf("No memory for \'mat\'!\n"); 
        exit(ERROR); 
    }
    initialMatrix(mat); 
    
    // 开始求解偏微分方程
    MPI_Init(&argc, &argv); 
    JacobiSequential(mat); 
    MPI_Finalize(); 
    
    // 输出偏微分方程结果到指定文件
    if(outputValues(mat->height, mat->width, mat->values) == 0) printf("The result is ouput to file \'%s\'. \n", OUTPUT_FILENAME); 
    // 释放内存
    freeMatrix(mat); 
    return 0; 
}

void JacobiSequential(Matrix mat)
{
    int k, i, j; 
    double*** temp; 
    int width = mat->width; 
    int height = mat->height; 
    int iter; 
    double max_delta, delta; 
    double time; 
    
    // 创建临时数组，存放上一次和当前的迭代结果
    temp = (double***)malloc(sizeof(double**)*2); 
    if(temp == NULL){
        printf("No memory for \'temp\'!\n"); 
        exit(ERROR); 
    }
    for(k = 0; k < 2; k++){
        temp[k] = (double**) malloc(sizeof(double*)*height); 
        if(temp[k] == NULL){
            printf("No memory for \'temp[%d]\'!\n", k); 
            exit(ERROR); 
        }
        for(i = 0; i < height; i++){
            temp[k][i] = (double*) malloc(sizeof(double)*width); 
            if(temp[k][i] == NULL){
                printf("No memory for \'temp[%d][%d]\'!\n", k, i); 
                exit(ERROR); 
            }
        }
    }
    // 开始求解偏微分方程，并计时
    time = MPI_Wtime(); 
    // temp[0]作为迭代的起始状态
    for(i = 0; i < height; i++)
        for(j = 0; j < width; j++)
            temp[0][i][j] = mat->values[i][j]; 
    // 初始化temp[1]的边界值
    for(j = 1; j < width-1; j++){
        temp[1][0][j] = temp[0][0][j]; 
        temp[1][height-1][j] = temp[0][height-1][j]; 
    }
    for(i = 0; i < height; i++){
        temp[1][i][0] = temp[0][i][0]; 
        temp[1][i][width-1] = temp[0][i][width-1]; 
    }
    
    // 开始迭代
    k = 0; 
    max_delta = eps+1.0; 
    for(iter = 0; iter < numIters && max_delta > eps; iter++){
        max_delta = 0.0; 
        // 通过偏微分方程公式，更新每个网格
        // 其中temp[1-k]是当前值，temp[k]是上一次迭代的值
        for(i = 1; i < height-1; i++)
            for(j = 1; j < width-1; j++){
                temp[1-k][i][j] = (temp[k][i-1][j] 
                                  +temp[k][i+1][j] 
                                  +temp[k][i][j-1]
                                  +temp[k][i][j+1])*0.25; 
                delta = fabs(temp[1-k][i][j] - temp[k][i][j]); 
                if(delta > max_delta)
                    max_delta = delta; 
            }
        k = 1 - k; 
        printf("iter: %d, max_delta = %lf; \n", iter, max_delta);   
    }
    
    // 将最终迭代结果放回矩阵mat中
    for(i = 1; i < height-1; i++)
        for(j = 1; j < width-1; j++)
            mat->values[i][j] = temp[k][i][j]; 
    // 结束计时，并输出所花的时间
    time = MPI_Wtime() - time; 
    printf("The time cost is %lf. \n", time); 
    // 计算并输出迭代结果与真实值之间的L2-norm
    printf("The L2norm of cost is %lf. \n", costL2norm(mat)); 
    
    // 释放temp的内存
    for(k = 0; k < 2; k++){
        for(i = 0; i < height; i++)
            free(temp[k][i]); 
        free(temp[k]); 
    }
    free(temp); 
    return ; 
}

double costL2norm(Matrix mat)
{
    double** predict = mat->values; 
    double cost = 0.0, diff; 
    int i, j; 
    
    // 根据公式计算得到相应的真实值，与预测值predict比对，得到L2-norm
    for(i = 0; i < mat->height; i++)
        for(j = 0; j < mat->width; j++){
            diff = fabs(predict[i][j] - sin(PI*j*step_x)*exp(-PI*i*step_y)); 
            cost += diff * diff; 
        }
    return sqrt(cost); 
}

// u(x, 0) = sin(PI*x)
// u(x, 1) = sin(PI*x)*e^(-x)
// u(0, y) = u(1, y) = 0
// analytical solution: 
// u(x, y) = sin(PI*x)*e^(-PI*y)
// range: 0 <= x,y <= 1
void initialMatrix(Matrix mat)
{
    int width = N; 
    int height = N; 
    double** values; 
    int i, j; 
    
    // 给values分配内存，存放矩阵的值
    values = (double**) malloc(sizeof(double*)*height); 
    if(values == NULL){
        printf("No memory for \'values\'!\n"); 
        exit(ERROR); 
    }
    for(i = 0; i < height; i++){
        values[i] = (double*) malloc(sizeof(double)*width); 
        if(values[i] == NULL){
            printf("No memory for \'values[%d]\'!\n", i); 
            exit(ERROR); 
        }
    }
    
    // 初始化网格矩阵的边界值
    for(j = 1; j < width-1; j++){
        values[0][j] = sin(PI * j * step_x); 
        values[height-1][j] = sin(PI * j * step_x)*exp(-j * step_x); 
    }
    for(i = 0; i < height; i++){
        values[i][0] = 0.0; 
        values[i][width-1] = 0.0; 
    }
    // 网格矩阵内部值全是0.0
    for(i = 1; i < height-1; i++)
        for(j = 1; j < width-1; j++)
            values[i][j] = 0.0; 
            
    mat->width = width; 
    mat->height = height; 
    mat->values = values; 
    return ; 
}

void freeMatrix(Matrix mat)
{
    int i; 
    for(i = 0; i < mat->height; i++)
        free(mat->values[i]); 
    free(mat->values); 
    free(mat); 
    return ; 
}

int outputValues(int height, int width, double** values)
{
    FILE *fp; 
    int i, j; 
    
    fp = fopen(OUTPUT_FILENAME, "w"); 
    if(fp == NULL){
        printf("Cannot open file \'%s\'!\n", OUTPUT_FILENAME); 
        return ERROR; 
    }
    for(i = 0; i < height; i++){
        for(j = 0; j < width; j++)
            fprintf(fp, "%8.3lf\t", values[i][j]); 
        fprintf(fp, "\n"); 
    }
    fclose(fp); 
    return 0; 
}
