#include "../include/dnn/dnn.hpp"
#include "../include/viennacl/tools/random.hpp"
#include <cmath>
#include <iostream>
using namespace std;

DNN::DNN(int L, int* n, Activation* activations, int batch_size)
{
    viennacl::tools::normal_random_numbers<ScalarType> random;

    this->L=L;
    this->n=n;
    this->activations=activations;
    this->batch_size=batch_size;
    this->W=new ScalarType*[L];
    for(int l=1;l<L;l++) 
    {
        W[l]=new ScalarType[(n[l-1]+1)*n[l]];
        for(int i=0;i<(n[l-1]+1)*n[l];i++) W[l][i]=random();
    }
    this->A=new ScalarType*[L];
    for(int l=0;l<L;l++)
    {
        A[l]=new ScalarType[batch_size*(n[l]+1)];
        for(int i=0;i<batch_size;i++) A[l][n[l]+i*(n[l]+1)]=1;
    }
    this->Z=new ScalarType*[L];
    for(int l=1;l<L;l++)
    {
        Z[l]=new ScalarType[batch_size*n[l]];
    }
    this->Delta=new ScalarType*[L];
    for(int l=1;l<L;l++)
    {
        Delta[l]=new ScalarType[batch_size*n[l]];
    }
}

DNN::~DNN()
{
    for(int l=1;l<L;l++) 
    {
        delete[] W[l];
        delete[] A[l];
        delete[] Z[l];
    }
    delete[] A[0];
    delete[] W;
    delete[] A;
    delete[] Z;
}

void DNN::activate(int l, int batch_size)
{
    switch(activations[l])
    {
    case SIGMOID:
        for(int i=0;i<batch_size;i++)
        {
            for(int j=0;j<n[l];j++)
            {
                A[l][i*(n[l]+1)+j]=1/(1+exp(-Z[l][i*n[l]+j]));
            }
        }
        break;
    case TANH:
        break;
    case RELU:
        break;
    }
}

void DNN::forward_propagation(ScalarType* X, ScalarType* Y, int batch_size)//m是样本个数，若大于batch_size则报错
{
    if(batch_size>this->batch_size) throw "样本个数超出缓存！";
    for(int i=0;i<batch_size;i++)
        for(int j=0;j<n[0];j++)
            A[0][i*(n[0]+1)+j]=X[i*n[0]+j];
    
    for(int l=1;l<L;l++)
    {
        for(int i=0;i<batch_size;i++)
        {
            for(int j=0;j<n[l];j++)
            {
                Z[l][i*n[l]+j]=0;
                for(int k=0;k<n[l-1]+1;k++)
                    Z[l][i*n[l]+j]+=A[l-1][i*(n[l-1]+1)+k]*W[l][k*n[l]+j];
            }
        }
        activate(l,batch_size);
    }

    for(int i=0;i<batch_size;i++)
        for(int j=0;j<n[L-1];j++)
            Y[i*n[L-1]+j]=A[L-1][i*(n[L-1]+1)+j];
}

void DNN::d_activate(int l, int batch_size)
{
    switch(activations[l])
    {
        case SIGMOID:
        {
            for(int i=0;i<batch_size;i++)
            {
                for(int j=0;j<n[l];j++)
                {
                    ScalarType tmp=1/(1+exp(-Z[l][i*n[l]+j]));
                    Delta[l][i*n[l]+j]=tmp*(1-tmp);
                }
            }
            break;
        }
        case TANH:
        {
            break;
        }
        case RELU:
        {
            break;
        }
    }
}
ScalarType DNN::d_activate(int l, ScalarType x)
{
    ScalarType result=0;
    switch(activations[l])
    {
        case SIGMOID:
        {
            ScalarType tmp=1/(1+exp(-x));
            result=tmp*(1-tmp);
            break;
        }
        case TANH:
        {
            break;
        }
        case RELU:
        {
            break;
        }
    }
    return result;
}

void DNN::back_propagation(ScalarType* Y, ScalarType* Label_Y, int batch_size, ScalarType eta)
{
    //计算输出层Delta
    for(int i=0;i<batch_size;i++)
    {
        for(int j=0;j<n[L-1];j++)
        {
            Delta[L-1][i*n[L-1]+j]=
                (Y[i*n[L-1]+j]-Label_Y[i*n[L-1]+j])
                *
                d_activate(L-1,Z[L-1][i*n[L-1]+j]);
        }
    }
    //计算各层Delta
    for(int l=L-2;l>0;l--)
    {
        for(int i=0;i<batch_size;i++)
        {
            for(int j=0;j<n[l];j++)
            {
                ScalarType tmp=0;
                for(int k=0;k<n[l+1];k++)
                {
                    tmp+=Delta[l+1][i*n[l+1]+k]*W[l+1][j*n[l+1]+k];
                }
                Delta[l][i*n[l]+j]=tmp*d_activate(l,Z[l][i*n[l]+j]);
            }
        }
    }
    //更新各层W
    for(int l=1;l<L;l++)
    {
        for(int i=0;i<n[l-1]+1;i++)
        {
            for(int j=0;j<n[l];j++)
            {
                for(int k=0;k<batch_size;k++)
                {
                    W[l][i*n[l]+j]-=eta*A[l-1][k*(n[l-1]+1)+i]*Delta[l][k*n[l]+j];
                }
            }
        }
    }
}

void DNN::train(ScalarType* X, ScalarType* Label_Y, int total, int batch_size, int Epoches, ScalarType eta)
{
    if(batch_size>this->batch_size) throw "batch_size设置过大，超出缓存！";
    ScalarType* Y=new ScalarType[this->batch_size*n[L-1]];
    for(int epoch=0;epoch<Epoches;epoch++)
    {
        cout<<"Epoch="<<epoch<<endl;
        for(int i=0;i<total/batch_size;i++)
        {
            forward_propagation(&X[i*batch_size*n[0]], Y, batch_size);
            back_propagation(Y, &Label_Y[i*batch_size*n[L-1]], batch_size, eta);
        }
        int remain=total%batch_size;
        if(remain>0)
        {
            forward_propagation(&X[(total-remain)*n[0]], Y, remain);
            back_propagation(Y, &Label_Y[(total-remain)*n[L-1]], remain, eta);
        }
    }
}