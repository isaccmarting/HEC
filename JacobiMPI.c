#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 如果内存不足，返回值是ERROR
#define ERROR -1
#define PI 3.1415926
// 偏微分方程结果的输出文件
#define OUTPUT_FILENAME "result.txt"

// 头进程编号
#define HEAD 0
// 消息传递tag，这是迭代过程中边界值的tag
#define LAST_TAG 5
#define NEXT_TAG 6
// 消息传递tag，这是合并结果过程中传递矩阵行数的tag
#define HEIGHT 7
// 消息传递tag，这是合并结果过程中传递矩阵每行值的tag，会随行数递增
// 'COLLECT' MUST be the BIGGEST number. 
#define COLLECT 8

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
// 求解偏微分方程的函数，这个是MPI并行方法
void JacobiMPI(Matrix mat); 
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
// myRank是当前进程的编号；numProcs是进程总数
int myRank, numProcs; 

int main(int argc, char* argv[])
{
    Matrix mat; 
    
    // 初始化设置
    N = 2500; numIters = 200; eps = 0.001; 
    step_x = step_y = 1.0 / (N-1); 
    
    // 初始化MPI并行，并得到进程总数和当前进程的编号
    MPI_Init(&argc, &argv); 
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs); 
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank); 
    
    // 初始化网格矩阵mat
    mat = (Matrix) malloc(sizeof(struct MatrixConfigure)); 
    if(mat == NULL){
        printf("No memory for \'mat\'!\n"); 
        exit(ERROR); 
    }
    initialMatrix(mat); 
    // 开始求解偏微分方程
    JacobiMPI(mat); 
    // 由HEAD进程收集最终的结果
    if(myRank == HEAD){
        int i, j; 
        int height, base; 
        double** full; 
        MPI_Status status; 
        
        // 给结果网格矩阵full分配内存
        full = (double**) malloc(sizeof(double)*N); 
        if(full == NULL){
            printf("No memory for \'full\'!\n"); 
            exit(ERROR); 
        }
        for(i = 0; i < N; i++){
            full[i] = (double*) malloc(sizeof(double)*N); 
            if(full[i] == NULL){
                printf("No memory for \'full[%d]\'!\n", i); 
                exit(ERROR); 
            }
        }
        
        // 把HEAD进程的结果放到矩阵full中
        for(i = 0; i < mat->height; i++)
            for(j = 0; j < mat->width; j++)
                full[i][j] = mat->values[i][j]; 
        // 进程结果开始的行号
        base = mat->height; 
        // 对每个进程接收它的迭代结果
        for(i = 1; i < numProcs; i++){
            // 首先接收进程i结果矩阵的行数
            MPI_Recv(&height, 1, MPI_INT, i, HEIGHT, MPI_COMM_WORLD, &status); 
            // 之后接收进程i结果矩阵的每行值
            for(j = 0; j < height; j++)
                MPI_Recv(full[base+j], N, MPI_DOUBLE, i, COLLECT+j, MPI_COMM_WORLD, &status); 
            base += height; 
        }
        // 输出偏微分方程结果到指定文件
        if(outputValues(N, N, full) == 0) printf("The result is ouput to file \'%s\'. \n", OUTPUT_FILENAME); 
        for(i = 0; i < N; i++)
            free(full[i]); 
        free(full); 
    }
    // 其它进程传递各自的结果到HEAD进程
    else{
        int j; 
        
        // 首先发送当前进程结果矩阵的行数
        MPI_Send(&(mat->height), 1, MPI_INT, HEAD, HEIGHT, MPI_COMM_WORLD); 
        // 之后发送当前进程结果矩阵的每行值
        for(j = 0; j < mat->height; j++)
            MPI_Send(mat->values[j], mat->width, MPI_DOUBLE, HEAD, COLLECT+j, MPI_COMM_WORLD); 
    }
    // 释放内存
    freeMatrix(mat); 
    MPI_Barrier(MPI_COMM_WORLD); 
    
    MPI_Finalize(); 
    return 0; 
}

void JacobiMPI(Matrix mat)
{
    int k, i, j; 
    double*** temp; 
    int width = mat->width; 
    int height = mat->height; 
    int tempHeight = height + 2; 
    int iter; 
    double max_delta, delta; 
    int next=myRank+1, last=myRank-1; 
    int start, end; 
    MPI_Status status; 
    double time; 
    
    // 创建临时数组，存放上一次和当前的迭代结果
    temp = (double***)malloc(sizeof(double**)*2); 
    if(temp == NULL){
        printf("No memory for \'temp\' at process \'%d\'!\n", myRank); 
        exit(ERROR); 
    }
    for(k = 0; k < 2; k++){
        temp[k] = (double**) malloc(sizeof(double*)*tempHeight); 
        if(temp[k] == NULL){
            printf("No memory for \'temp[%d]\' at process \'%d\'!\n", k, myRank); 
            exit(ERROR); 
        }
        for(i = 0; i < tempHeight; i++){
            temp[k][i] = (double*) malloc(sizeof(double)*width); 
            if(temp[k][i] == NULL){
                printf("No memory for \'temp[%d][%d]\' at process \'%d\'!\n", k, i, myRank); 
                exit(ERROR); 
            }
        }
    }
    // 开始求解偏微分方程，并计时
    MPI_Barrier(MPI_COMM_WORLD); 
    time = MPI_Wtime(); 
    // 注意，这里的temp和串行方法的不一样
    // 这里的矩阵temp，第一行存放上一个矩阵区域的最后一行结果，最后一行存放下一个矩阵区域的第一行结果
    // temp[0]作为迭代的起始状态
    for(i = 1; i < tempHeight-1; i++)
        for(j = 0; j < width; j++)
            temp[0][i][j] = mat->values[i-1][j]; 
    // 初始化temp[1]的边界值
    for(j = 1; j < width-1; j++){
        temp[1][1][j] = temp[0][1][j]; 
        temp[1][tempHeight-2][j] = temp[0][tempHeight-2][j]; 
    }
    for(i = 1; i < tempHeight-1; i++){
        temp[1][i][0] = temp[0][i][0]; 
        temp[1][i][width-1] = temp[0][i][width-1]; 
    }
    
    // 根据进程的id，设置更新起始的行号和终止的行号
    // 如果是进程HEAD，更新需要从temp[2]开始，否则temp[1]
    if(myRank == HEAD) start = 2; 
    else start = 1; 
    // 如果是倒数第一个进程，更新需要到倒数第二行结束，否则倒数第一行
    if(myRank == numProcs-1) end = tempHeight-2; 
    else end = tempHeight-1; 
    // 开始迭代
    k = 0; 
    max_delta = eps+1.0; 
    for(iter = 0; iter < numIters && max_delta > eps; iter++){
        // 更新进程的矩阵区域的边界值，从相应进程获取上一轮中它们矩阵区域最后一行和第一行的结果
        if(last >= 0)
            MPI_Sendrecv(temp[k][1], width, MPI_DOUBLE, last, LAST_TAG, 
                         temp[k][0], width, MPI_DOUBLE, last, NEXT_TAG, 
                         MPI_COMM_WORLD, &status); 
        if(next < numProcs)
            MPI_Sendrecv(temp[k][tempHeight-2], width, MPI_DOUBLE, next, NEXT_TAG, 
                         temp[k][tempHeight-1], width, MPI_DOUBLE, next, LAST_TAG, 
                         MPI_COMM_WORLD, &status); 
        max_delta = 0.0; 
        // 通过偏微分方程公式，更新每个网格
        // 其中temp[1-k]是当前值，temp[k]是上一次迭代的值
        for(i = start; i < end; i++)
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
        // 结合所有进程的max_delta，更新全局的max_delta
        delta = max_delta; 
        MPI_Allreduce(&delta, &max_delta, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); 
        if(myRank == HEAD)
            printf("iter: %d, max_delta = %lf; \n", iter, max_delta); 
    }
    
    // 将最终迭代结果放回矩阵mat中
    // i is from 1 to tempHeight-1，去除存放边界值的行
    for(i = 1; i < tempHeight-1; i++)
        for(j = 1; j < width-1; j++)
            mat->values[i-1][j] = temp[k][i][j]; 
    // 结束计时，并由进程HEAD输出所花的时间
    MPI_Barrier(MPI_COMM_WORLD); 
    time = MPI_Wtime() - time; 
    if(myRank == HEAD)
        printf("The time cost is %lf. \n", time); 
    
    // 释放temp的内存
    for(k = 0; k < 2; k++){
        for(i = 0; i < height; i++)
            free(temp[k][i]); 
        free(temp[k]); 
    }
    free(temp); 
    return ; 
}

// Matrix 'mat' must be a full matrix. 
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
    int height; 
    double** values; 
    int i, j; 
    
    // 根据线程数划分网格矩阵，按带均匀分割，多出的行均匀分到开始的几个矩阵区域
    if(myRank < N % numProcs)
        height = N / numProcs + 1; 
    else
        height = N / numProcs; 
    // 给values分配内存，存放矩阵的值
    values = (double**) malloc(sizeof(double*)*height); 
    if(values == NULL){
        printf("No memory for \'values\' at process \'%d\'!\n", myRank); 
        exit(ERROR); 
    }
    for(i = 0; i < height; i++){
        values[i] = (double*) malloc(sizeof(double)*width); 
        if(values[i] == NULL){
            printf("No memory for \'values[%d]\' at process \'%d\'!\n", i, myRank); 
            exit(ERROR); 
        }
    }
    
    // 初始化网格矩阵的值
    for(i = 0; i < height; i++)
        for(j = 0; j < width; j++)
            values[i][j] = 0.0; 
    // 根据进程编号初始化边界值
    // 进程HEAD需要初始化第一行
    if(myRank == HEAD)
        for(j = 1; j < width-1; j++)
            values[0][j] = sin(PI * j * step_x); 
    // 最后一个进程需要初始化最后一行
    if(myRank == numProcs-1)
        for(j = 1; j < width-1; j++)
            values[height-1][j] = sin(PI * j * step_x)*exp(-j * step_x); 
    
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
