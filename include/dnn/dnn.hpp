#ifndef DNNWU
#define DNNWU
#include "../include/viennacl/tools/random.hpp"

typedef float ScalarType;

enum Activation
{
    SIGMOID,
    TANH,
    RELU
};

class DNN
{
public:
    DNN(int L, int* n, Activation* activations,int batch_size=1024);
    ~DNN();
    void forward_propagation(ScalarType* X, ScalarType* Y, int batch_size=1);//batch_size是样本个数
    void train(ScalarType* X, ScalarType* Label_Y, int total, int batch_size, int Epoches, ScalarType eta=0.01);
private:
    int L;
    int* n;
    Activation* activations;
    int batch_size;
    ScalarType** W;
    ScalarType** A;
    ScalarType** Z;
    ScalarType** Delta;

    void activate(int l, int batch_size=1);//第l层，有batch_size个样本
    void d_activate(int l, int batch_size);
    ScalarType d_activate(int l, ScalarType x);
    //Y是前向传播的输出，Label_Y是标签，batch_size是样本个数，eta是训练步长
    void back_propagation(ScalarType* Y, ScalarType* Label_Y, int batch_size, ScalarType eta);
};


#endif