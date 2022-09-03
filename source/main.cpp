#include "../include/load_MNIST.hpp" 
#include <iostream>
#include <opencv2/opencv.hpp>
#include "../include/dnn/dnn.hpp"
using namespace std;

int main()
{
    string train_image_path="resource/train-images-idx3-ubyte";
    string train_label_path="resource/train-labels-idx1-ubyte";

    cv::Mat train_image=ReadImages(train_image_path);
    cv::Mat train_label=ReadLabels(train_label_path);
    for(int i=0;i<10;i++) cout<<int(train_label.at<uchar>(i,0))<<"  ";
    cout<<endl;
    show100Images(train_image);

    ScalarType* X=new ScalarType[1000*28*28];
    ScalarType* Label_Y=new ScalarType[1000*10];
    for(int i=0;i<1000;i++)
    {
        for(int j=0;j<28*28;j++)
        {
            X[i*28*28+j]=((ScalarType)train_image.data[i*28*28+j])/255;
        }
        for(int j=0;j<10;j++)
        {
            Label_Y[i*10+j]=0;
        }
        Label_Y[i*10+(int)train_label.data[i]]=1;
    } 
    int n[3]={28*28,200,10};
    Activation activations[3]={SIGMOID,SIGMOID,SIGMOID};
    DNN mydnn(3,n,activations);
    mydnn.train(X,Label_Y,1000,100,100);
    ScalarType* Y=new ScalarType[1000*10];
    mydnn.forward_propagation(X,Y,1000);
    for(int i=0;i<10;i++)
    {
        ScalarType tmp=0;
        int max=-1;
        for(int j=0;j<10;j++)
        {
            if(Y[i*10+j]>tmp)
            {
                tmp=Y[i*10+j];
                max=j;
            }
        }
        cout<<max<<"  ";
    }
    return 0;
}