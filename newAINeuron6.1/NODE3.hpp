//
//  NODE.hpp
//  newAINeuron2
//
//  Created by Eric Cheng on 7/19/18.
//  Copyright © 2018 Eric Cheng. All rights reserved.
//

#ifndef NODE3_hpp
#define NODE3_hpp


#define outGoingEdge 6

#define sigmoid 2
#define relu 1//there might be some bug with relu
#define Tanh 0

int NodeCount = 0;
double ALPHA = 0.0003;
double Beta = 0.99;
double T = 1.0;
#define lowerThreshold 0.95

#include <stdio.h>
#include <time.h>
#include <cstdio>
#include <vector>
#include <iostream>
#include <queue>
#include <cmath>
#include <random>
using namespace std;
std::default_random_engine generator;
std::normal_distribution<double> distribution(0.0,1.0);
std::normal_distribution<double> distribution2(0.0,1.0);

struct coordAndCount {
    int xMean = 0, yMean = 0, countT = 0;
};

class node{
public:
    vector<node*> backwardConnection;
    vector<node*> forwardConnection;
    int ID;
    int calSeq = 0;
    bool inferenced = false;
    double delta = 0;
    coordAndCount CAC;

    
    void clearMomentum(){
        for (int i=0; i<d.size(); i++) {
            d[i] = 0;
        }
    }
    
    int forwardConnectionSize(){
        return forwardConnection.size();
    }
    node(int type, int ID, int inputCount){
        //bias = (distribution2(generator));
        output = 0;
        this->ID = ID;
        if(rand()%2==0) activationType = rand()%1;
        else activationType = rand()%1;
        nodeType = type;
        if (type == 0) {
            CAC.xMean = inputCount/28;
            CAC.yMean = inputCount%28;
            CAC.countT = 1;
        }
    }
    void perturb(){
        for (int i=0; i<weight.size(); i++) {
            double tmpWeight = weight[i]*1.025;
            if (rand()%2==0 && abs(tmpWeight)<=10) {
                weight[i] = tmpWeight;
            }
            else{
                weight[i] = weight[i]/1.025;
            }
        }
    }
    int weightSize(){
        return weight.size();
    }
    void clearCalSeq(){
        calSeq = 0;
    }
    
    void connect(node* x){
        NodeCount++;
        this->forwardConnection.push_back(x);
        x->weight.push_back(sqrt(this->forwardConnectionSize() + x->backwardConnection.size() + 0.5) * distribution(generator));
        x->inputBarrier.push_back(abs(double(distribution2(generator))));
        x->backwardConnection.push_back(this);
        x->d.push_back(0.0);
        
        x->CAC.xMean = (x->CAC.xMean*x->CAC.countT + this->CAC.xMean*this->CAC.countT)/(x->CAC.countT+this->CAC.countT);
        x->CAC.yMean = (x->CAC.yMean*x->CAC.countT + this->CAC.yMean*this->CAC.countT)/(x->CAC.countT+this->CAC.countT);
        x->CAC.countT += 1;
    }

    
    void setInput(double x){
        output = x;
    }
    void calculate(bool isTraining){
        inferenced = true;
        double tmpOut = sumOfInput();
        //double inputExceedBarrier = oneInputExceedBarrier(isTraining);
        //if(inputExceedBarrier-0.00000001>=0) {
        //    output = inputExceedBarrier;

        //}
        //else{
            switch (activationType) {
                case sigmoid://
                    output = 1.0/(1.0+exp(-tmpOut));
                    backProp = -1;
                    break;
                case relu:
                    output = (tmpOut>=0)? tmpOut: 0.01*tmpOut;
                    backProp = -1;
                    break;
               
                case Tanh:
                    output = tanh(tmpOut);
                    backProp = -1;
                    break;
                default:
                    output = tmpOut;
                    backProp -1;
                    break;
            }
        //}
    }
    double connectionSum(){
        double sum = 0;
        int maxIndex = min(backwardConnection.size(), weight.size());
        for (int i=0; i<maxIndex; i++) {
            if(backwardConnection[i]==NULL) continue;
            sum = sum + backwardConnection[i]->outVal()*weight[i];
        }
        return sum;
    }
    double getVal(){
        return sumOfRawInput();
    }
    void setDelta(double d){
        this->delta = d + this->delta;
    }
    void clearDelta(){
        this->delta = 0;
    }
    void update(bool isOutput, double sign){//backprop
        t+=0.1;
        if (isOutput) this->delta = sign;//(sign<0)? -1.0 : +1.0;
        int maxIndex = min(backwardConnection.size(), weight.size());
        for (int i=0; i<maxIndex; i++) {
            double div;
            switch (activationType) {
                case sigmoid:
                    div = (1.0/(1+exp(-sumOfRawInput()))) * (1-1.0/(1+exp(-sumOfRawInput())));
                    break;
                case relu:
                    div = (output>=0)? 1.0 : 0.01;
                    break;
                case Tanh:
                    div = 1-tanh(sumOfRawInput())*tanh(sumOfRawInput());
                    break;
                default:
                    div = 0;
                    break;
            }
            if (backProp!=-1) {
                div = 0;
                i = backProp;
            }
            
            
            this->backwardConnection[i]->setDelta(delta*weight[i]);
            d[i] = -ALPHA * this->delta * this->backwardConnection[i]->outVal()*div + Beta*d[i];
            if (!isnan(d[i])) {
                //double tmpW = weight[i] + d[i]/((double)(inputCount + 0.5));
                double tmpW = weight[i]*0.9995 + d[i];
                if(abs(tmpW)>10) weight[i] = abs(weight[i])/weight[i] * 10;
                else weight[i] = tmpW;
            }
            else{
                d[i] = 0;
            }
            
            if (backProp!=-1) break;
        }
    }
    vector<node*> getBackwardConnection(){
        return backwardConnection;
    }
    double outVal(){
        return output ;
    }
private:
    double bias;
    double output;
    int backProp = -1;
    int activationType; //0-->simoid, 1-->relu, 2-->step,
    int inputCount = 0;
    int nodeType; //0->input 1->mid 2->out
    double t = 1;
    vector<double> d;
    vector<double> weight;
    vector<double> inputBarrier;
    
    
    
    
    double sumOfInput(){
        double sum = 0;
        inputCount = 0;
        int maxIndex = min(backwardConnection.size(), weight.size());
        for (int i=0; i<maxIndex; i++) {
            if(backwardConnection[i]==NULL) continue;
            inputCount++;
            sum = sum + backwardConnection[i]->outVal()*weight[i];
            if (isnan(weight[i])) {
                //printf("nan\n");
                return 0;
            }
        }
        //return (inputCount==0)? 0 : sum/(double)inputCount;
        return sum;
    }
    
    
    double sumOfRawInput(){
        double sum = 0;
        int maxIndex = min(backwardConnection.size(), weight.size());
        inputCount = 0;
        for (int i=0; i<maxIndex; i++) {
            if(backwardConnection[i]==NULL) continue;
            //inputCount++;
            sum = sum + backwardConnection[i]->outVal()*weight[i];
        }
        //return (inputCount==0)? 0 : sum/(double)inputCount;
        return sum;
    }
    
    double oneInputExceedBarrier(bool isTraining){
        double max = 0;
        backProp = -1;
        for (int i=0; i<weight.size(); i++) {
            if (inputBarrier[i] < weight[i]*this->backwardConnection[i]->outVal()) {
                if (weight[i]*this->backwardConnection[i]->outVal() > max) {
                    max = weight[i]*this->backwardConnection[i]->outVal();
                    backProp = i;
                }
                //return weight[i]*this->backwardConnection[i]->outVal();
            }
        }
        return (max==0)?0 : 1;
    }
    
    double sumOfBarrier(){
        double tmp = 0;
        int indexMax = min(inputBarrier.size(), this->backwardConnection.size());
        for (int i=0; i<indexMax; i++) {
            tmp = tmp + inputBarrier[i];
        }
        return tmp;
    }
};
#endif /* NODE_hpp */
