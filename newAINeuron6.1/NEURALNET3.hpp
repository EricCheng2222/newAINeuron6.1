//
//  NEURALNET.hpp
//  newAINeuron2
//
//  Created by Eric Cheng on 7/19/18.
//  Copyright © 2018 Eric Cheng. All rights reserved.
//

#ifndef NEURALNET_hpp
#define NEURALNET_hpp

#include "NODE3.hpp"
#include <cmath>
int const trainingTime = 1000;
int inverseAddNodeProb = 1;
double preLoss=100000000;
extern double T;
extern int NodeCount;

/*
double confidenceRate(){
    return 6.4/pow(trainingTime, 4)*pow(T-trainingTime/2, 4) + 0.8; // y = ax^2 + b and bound y (0.8, 1.2)
}
 */


void selectionSort(vector<node*> &arr, int n)
{
    int i, j, min_idx;
    for (i = 0; i < n-1; i++)
    {
        min_idx = i;
        for (j = i+1; j < n; j++)
            if (arr[j]->calSeq > arr[min_idx]->calSeq)
                min_idx = j;
        swap(arr[min_idx], arr[i]);
    }
}

bool inRTN(int a, vector<int>&x){
    for (int i=0; i<x.size(); i++) {
        if (x[i]==a) {
            return true;
        }
    }
    return false;
}




//BUG but useable
vector<int> sortNodesByDelta(vector<node*> &arr)
{
    vector<int> rtn;
    
    int i, j, min_idx;
    for (i = 0; i < 5; i++)
    {
        min_idx = i;
        for (j = i+1; j < arr.size(); j++)
            if (abs(arr[j]->delta) > abs(arr[min_idx]->delta) && !inRTN(j, rtn)){
                min_idx = j;
            }
        swap(arr[i], arr[min_idx]);
        rtn.push_back(min_idx);
    }
    return rtn;
}



void dfsCalSeq(vector<node*> &firstNode, int searchDirection){//search = 0 forForward。=1 forBackward
    vector<node*> stk;
    vector<node*> calculated;
    for(int i=0; i<firstNode.size(); i++)stk.push_back(firstNode[i]);
    for(int i=0; i<firstNode.size(); i++)firstNode[i]->calSeq = 0;
    
    if (searchDirection==0) {
        while (stk.empty()==false) {
            node* tmp = stk.back();
            int tmpSeq = tmp->calSeq + 1;
            stk.pop_back();
            for (int i=0; i<tmp->forwardConnection.size(); i++) {
                if(tmp->forwardConnection[i]==NULL) continue;
                stk.push_back(tmp->forwardConnection[i]);
                if(tmp->forwardConnection[i]->calSeq < tmpSeq){
                    tmp->forwardConnection[i]->calSeq = tmpSeq;
                }
            }
        }
    }
    else{
        while (stk.empty()==false) {
            node* tmp = stk.back();
            int tmpSeq = tmp->calSeq + 1;
            stk.pop_back();
            for (int i=0; i<tmp->backwardConnection.size(); i++) {
                if(tmp->backwardConnection[i]==NULL) continue;
                stk.push_back(tmp->backwardConnection[i]);
                if(tmp->backwardConnection[i]->calSeq < tmpSeq){
                    tmp->backwardConnection[i]->calSeq = tmpSeq;
                }
            }
        }
    }
}

void infer(vector<node*> &firstNode, bool isTraining){
    vector<node*> toBeCalculate;
    vector<node*> calculated;
    vector <node*> q;
    for(int i=0; i<firstNode.size(); i++) q.push_back(firstNode[i]);
    while (q.empty()==false) {
        node* tmp = q.back();
        q.pop_back();
        if (q.empty()==true) {
            selectionSort(toBeCalculate, toBeCalculate.size());
            q = toBeCalculate;
            toBeCalculate.clear();
        }
        bool isFirstNode = false;
        for (int i=0; i<firstNode.size(); i++) {
            if(tmp==firstNode[i]){
                isFirstNode = true;
                break;
            }
        }
        if (isFirstNode == false) {
            tmp->calculate(isTraining);
        }
        calculated.push_back(tmp);
        for (int i=0; i<tmp->forwardConnection.size(); i++) {
            bool isCal = false;
            for (int j=0; j<calculated.size(); j++) {
                if (tmp->forwardConnection[i] == calculated[j]) {
                    isCal = true;
                    break;
                }
            }
            for (int j=0; j<q.size(); j++) {
                if (tmp->forwardConnection[i] == q[j]) {
                    isCal = true;
                    break;
                }
            }
            for (int j=0; j<toBeCalculate.size(); j++) {
                if (tmp->forwardConnection[i] == toBeCalculate[j]) {
                    isCal = true;
                    break;
                }
            }
            if (isCal==false) toBeCalculate.push_back(tmp->forwardConnection[i]);
        }
    }
}

void calculateBackProp(vector<node*> &firstNode, vector<double> &y){
    vector<double> predictY;
    for (int i=0; i<firstNode.size(); i++) {
        predictY.push_back(firstNode[i]->getVal());
    }
    
    //double sign = (y - predictY)/abs(y-predictY);
    vector<double> sign;
    //printf("%d\n", firstNode.size());
    double softSum = 0;
    for (int i=0; i<firstNode.size(); i++) {
        softSum = softSum + exp(predictY[i]);
    }
    
    
    double tmp;
    for (int i=0; i<firstNode.size(); i++) {
        
        tmp = exp(predictY[i])/softSum-y[i];
        sign.push_back(tmp);
        
    }
    
    vector<node*> q;
    vector<node*> calculated;
    vector<node*> toBeCalculate;

    q.clear();
    for(int i=0; i<firstNode.size(); i++) q.push_back(firstNode[i]);
    while (q.empty()==false) {
        node* tmp = q.back();
        q.pop_back();
        if (q.empty()==true) {
            selectionSort(toBeCalculate, toBeCalculate.size());
            q = toBeCalculate;
            toBeCalculate.clear();
        }
        
        bool isFirstNode = false;
        for (int i=0; i<firstNode.size(); i++) {
            if (tmp==firstNode[i]) {
                //printf("%x\n", tmp);
                tmp->update(1, sign[i]);
                isFirstNode = true;
                break;
            }
        }
        if(isFirstNode==false) {
            //printf("%x\n", tmp);
            tmp->update(0, 0);
        }
        
        calculated.push_back(tmp);
        for (int i=0; i<tmp->backwardConnection.size(); i++) {
            bool isCal = false;
            for (int j=0; j<calculated.size(); j++) {
                if (tmp->backwardConnection[i] == calculated[j]) {
                    isCal = true;
                    break;
                }
            }
            for (int j=0; j<q.size(); j++) {
                if (tmp->backwardConnection[i] == q[j]) {
                    isCal = true;
                    break;
                }
            }
            for (int j=0; j<toBeCalculate.size(); j++) {
                if (tmp->backwardConnection[i] == toBeCalculate[j]) {
                    isCal = true;
                    break;
                }
            }
            if (isCal==false) toBeCalculate.push_back(tmp->backwardConnection[i]);
        }
    }
}
    

struct nodeToWeightDic {
    nodeToWeightDic(double w, node *f, node* b){
        weight = w;
        frontNode = f;
        backNode = b;
    }
    nodeToWeightDic(){
        
    }
    double weight;
    node *frontNode;
    node *backNode;
};

bool cmpDIC(nodeToWeightDic &a, nodeToWeightDic &b){
    return abs(a.weight)<abs(b.weight);
}

class NeuralNet{
public:
    NeuralNet(int inputNode, int outputNode, int nodesMid){
        int inputOutputNodeType = 0;
        int midLayerNodeType = 1;
        int outLayerNodeType = 2;
        for (int i=0; i<inputNode; i++) inputLayer.push_back(new node(inputOutputNodeType, 0, i));//i-->input[i/28][i%28]
        nodeIDcount++;
        for (int i=0; i<outputNode; i++) {//PUTTING OUTPUT NODE HERE IS NOT A MISTTAKE, INITIALIZE NUMBER OF MID-NODE COUNT TO OUTPUT-NODE COUNT
            int setID = 0;
            if(rand()%1==0) setID = rand()%nodeIDcount + 1;//widening the network
            else setID = nodeIDcount++;//deepening the network
            midLayer.push_back(new node(midLayerNodeType, setID, 0));//0-->dummy, useless for mid
        }
        nodeIDcount++;
        for (int i=0; i<outputNode; i++) outputLayer.push_back(new node(outLayerNodeType, 200, 0));//SETTING 100 MEANS THE DEEPEST LAYER WONT EXCEED 100, 0-->useless for  output
        
        connectInit();//one input-node should at least connect to one mid-node/out-node and one mid node should at least connect to one output node
    }
    void Perturb(){
        for (int i=0; i<midLayer.size(); i++) {
            midLayer[i]->perturb();
        }
        for (int i=0; i<outputLayer.size(); i++) {
            outputLayer[i]->perturb();
        }
    }
    void train(vector<double> &x, vector<double> &y){
        T += 0.1;
        //inference() will dfscalseq & infer
        inference(x, true);//true --> training phase

        //set reverse dfs cal-seq for backprop
        dfsCalSeq(outputLayer, 1); //1 for backward
        
        //calculate backprop
        calculateBackProp(outputLayer, y);
    
        //clear delta and cal-seq
        for(int i=0; i<outputLayer.size(); i++) outputLayer[i]->clearDelta();
        for(int i=0; i<inputLayer.size(); i++) inputLayer[i]->clearDelta();
        for(int i=0; i<midLayer.size(); i++) midLayer[i]->clearDelta();
        for (int i=0; i<inputLayer.size(); i++) inputLayer[i]->clearCalSeq();
        for (int i=0; i<midLayer.size(); i++) midLayer[i]->clearCalSeq();
        for (int i=0; i<outputLayer.size(); i++) outputLayer[i]->clearCalSeq();
    }
    
    vector<double> inference(vector<double> &x, bool isTraining){
        for(int i=0; i<inputLayer.size(); i++) {
            inputLayer[i]->setInput(x[i]);
        }
        
        //dfs to set cal seq
        //dfsCalSeq(inputLayer[0], 0);//0 for forward
        //infer the neural net
        dfsCalSeq(inputLayer, 0);
        infer(inputLayer, isTraining); //inputLayer[0] is the starting node
        //clear cal-seq
        for (int i=0; i<inputLayer.size(); i++) inputLayer[i]->clearCalSeq();
        for (int i=0; i<midLayer.size(); i++) midLayer[i]->clearCalSeq();
        for (int i=0; i<outputLayer.size(); i++) outputLayer[i]->clearCalSeq();
        
        //get output and set sign
        vector<double> predictY;
        for(int i=0; i<outputLayer.size(); i++) predictY.push_back(outputLayer[i]->getVal());
        //for(int i=0; i<outputLayer.size(); i++) printf("trainInfr: %lf\n", predictY[i]);
        return predictY;
    }
    
    double calculateLoss(vector<double> &x, vector<double> &y){
        double sumOfLoss = 0;
        vector<double> infX = inference(x, false);//false stand for not Training phase
        double softSum = 0;
        for (int i=0; i<y.size(); i++){
            softSum += exp(infX[i]);
        }
        for (int i=0; i<y.size(); i++) {
            sumOfLoss += pow(y[i]-exp(infX[i])/softSum, 2);
        }
        return sumOfLoss;
    }
    
    void addNode(vector<double> &x, vector<double> &y){
        srand (time(NULL));
        int setID = 0;
        int midLayerNodeType = 1;
        bool didAddCount;
        vector<int> sequence;
        //inference() will dfscalseq & infer
        inference(x, false); // false --> not training phase
        //set reverse dfs cal-seq for backprop
        dfsCalSeq(outputLayer, 1); //1 for backward
        //calculate backprop
        calculateBackProp(outputLayer, y);
        vector<int> midOldIte = sortNodesByDelta(midLayer);
        vector<int> inputIte = sortNodesByDelta(inputLayer);
        vector<int> outputIte = sortNodesByDelta(outputLayer);
        
        if(rand()%3!=0 || nodeIDcount==199) {
            setID = rand()%nodeIDcount + 1;
            didAddCount = false;
        }
        else{
            setID = ++nodeIDcount;
            didAddCount = true;
        }
        midLayer.push_back(new node(midLayerNodeType, setID, 0));//0-->dummy value for midLayer
        bool tooManyNodes = midLayer.size()>1000;
        
        
        
        //input to new node
        int ite;
        ite = rand()%2;
        for (int i=0; i<ite; i++) {
            int mid = inputIte[0];
            int midI = mid/28;
            int midJ = mid%28;
            inputLayer[mid]->connect(midLayer[midLayer.size()-1]);
            for (int i=0; i<8; i++) {
                if(rand()%(i+1)==0 && isInBound(midI+i-1, midJ+i-1))
                    inputLayer[(midI+i-1)*28 + midJ+i-1]->connect(midLayer[midLayer.size()-1]);
                if(rand()%(i+1)==0 && isInBound(midI+i,   midJ+i-1))
                    inputLayer[(midI+i)*28 + midJ+i-1]->connect(midLayer[midLayer.size()-1]);
                if(rand()%(i+1)==0 && isInBound(midI+i+1, midJ+i-1))
                    inputLayer[(midI+i+1)*28 + midJ+i-1]->connect(midLayer[midLayer.size()-1]);
                
                if(rand()%(i+1)==0 && isInBound(midI+i-1, midJ+i))
                    inputLayer[(midI+i-1)*28 + midJ+i]->connect(midLayer[midLayer.size()-1]);
                if(rand()%(i+1)==0 && isInBound(midI+i+1, midJ+i))
                    inputLayer[(midI+i+1)*28 + midJ+i]->connect(midLayer[midLayer.size()-1]);
                
                if(rand()%(i+1)==0 && isInBound(midI+i-1, midJ+i+1))
                    inputLayer[(midI+i-1)*28 + midJ+i+1]->connect(midLayer[midLayer.size()-1]);
                if(rand()%(i+1)==0 && isInBound(midI+i,   midJ+i+1))
                    inputLayer[(midI+i)*28 + midJ+i+1]->connect(midLayer[midLayer.size()-1]);
                if(rand()%(i+1)==0 && isInBound(midI+i+1, midJ+i+1))
                    inputLayer[(midI+i+1)*28 + midJ+i+1]->connect(midLayer[midLayer.size()-1]);
            }
        }
        
        //midOld --> new node
        vector<node*> lowDis;
        
        //add four node that is close to highest delta
        int tmpS = rand()%3;
        for (int i=0; i<tmpS; i++) {
            ite = midOldIte[i];
            lowDis.clear();
            lowDis = nodesNearTo(midLayer[ite]);
            if (!tooManyNodes && !isLoop(midLayer[ite], midLayer[midLayer.size()-1])) {
                midLayer[ite]->connect(midLayer[midLayer.size()-1]);
            }
            for (int j=0; j<lowDis.size(); j++) {
                if (!tooManyNodes && !isLoop(lowDis[j], midLayer[midLayer.size()-1])) {
                    lowDis[j]->connect(midLayer[midLayer.size()-1]);
                }
            }
        }
        
        tmpS = rand()%3;
        //new node to mid old
        for (int i=0; i<tmpS; i++) {
            ite = midOldIte[i];
            lowDis.clear();
            lowDis = nodesNearTo(midLayer[ite]);
            if (!tooManyNodes && !isLoop(midLayer[ite], midLayer[midLayer.size()-1])) {
                midLayer[ite]->connect(midLayer[midLayer.size()-1]);
            }
            for (int j=0; j<lowDis.size(); j++) {
                if (!tooManyNodes && !isLoop(lowDis[j], midLayer[midLayer.size()-1])) {
                    lowDis[j]->connect(midLayer[midLayer.size()-1]);
                }
            }
        }
        
        tmpS = rand()%5;
        //new node to output
        for (int i=0; i<tmpS; i++) {
            ite = outputIte[i];
            if (!tooManyNodes && !isLoop(midLayer[midLayer.size()-1], outputLayer[ite])) {
                midLayer[midLayer.size()-1]->connect(outputLayer[ite]);
            }
        }
        
        if (midLayer[midLayer.size()-1]->forwardConnectionSize()==0 || midLayer[midLayer.size()-1]->backwardConnection.size()==0) {
            midLayer.pop_back();
            if(didAddCount==true)nodeIDcount--;
        }
         
        for(int i=0; i<outputLayer.size(); i++) outputLayer[i]->clearDelta();
        for(int i=0; i<inputLayer.size(); i++) inputLayer[i]->clearDelta();
        for(int i=0; i<midLayer.size(); i++) midLayer[i]->clearDelta();
        for (int i=0; i<inputLayer.size(); i++) inputLayer[i]->clearCalSeq();
        for (int i=0; i<midLayer.size(); i++) midLayer[i]->clearCalSeq();
        for (int i=0; i<outputLayer.size(); i++) outputLayer[i]->clearCalSeq();
    }
    
    void clearMomentum(){
        for(int i=0; i<inputLayer.size(); i++) inputLayer[i]->clearMomentum();
        for(int i=0; i<midLayer.size(); i++) midLayer[i]->clearMomentum();
        for(int i=0; i<outputLayer.size(); i++) outputLayer[i]->clearMomentum();

    }
    /*
    void kill(float percentage){
        for (int i=0; i<midLayer.size(); i++) {
            if(midLayer[i]->connectionCount()<killThreshold || midLayer[i]->connectionSum()<killThreshSum)
                //one cannot simply erase a node, the connection need to be handled
                //remove the these comments when finish this TODO
                midLayer.erase(midLayer.begin() + i);
        }
    }
    */
    bool isInBound(int x, int y){
        if (x<0 || x>27) return false;
        if (y<0 || y>27) return false;
        return true;
    }
    void connectInit(){
        //input ----> midlayer
        srand (time(NULL));
        bool larger = inputLayer.size() > midLayer.size();
        int connectNode;
        if (larger == true){
            int TMP = 0;
            for (int i=0; i<inputLayer.size()/70; i++) {
                int mid = rand()%(28*28);
                int midI = mid/28;
                int midJ = mid%28;
                
                int connectMid = rand()%midLayer.size();
                inputLayer[mid]->connect(midLayer[TMP%midLayer.size()]);
                TMP++;
                for (int i=0; i<8; i++) {
                    if(rand()%(i+1)==0 && isInBound(midI+i-1, midJ+i-1)) inputLayer[(midI+i-1)*28 + midJ+i-1]->connect(midLayer[connectMid]);
                    if(rand()%(i+1)==0 && isInBound(midI+i,   midJ+i-1)) inputLayer[(midI+i)*28 + midJ+i-1]->connect(midLayer[connectMid]);
                    if(rand()%(i+1)==0 && isInBound(midI+i+1, midJ+i-1)) inputLayer[(midI+i+1)*28 + midJ+i-1]->connect(midLayer[connectMid]);
                    
                    if(rand()%(i+1)==0 && isInBound(midI+i-1, midJ+i)) inputLayer[(midI+i-1)*28 + midJ+i]->connect(midLayer[connectMid]);
                    if(rand()%(i+1)==0 && isInBound(midI+i+1, midJ+i)) inputLayer[(midI+i+1)*28 + midJ+i]->connect(midLayer[connectMid]);
                    
                    if(rand()%(i+1)==0 && isInBound(midI+i-1, midJ+i+1)) inputLayer[(midI+i-1)*28 + midJ+i+1]->connect(midLayer[connectMid]);
                    if(rand()%(i+1)==0 && isInBound(midI+i,   midJ+i+1)) inputLayer[(midI+i)*28 + midJ+i+1]->connect(midLayer[connectMid]);
                    if(rand()%(i+1)==0 && isInBound(midI+i+1, midJ+i+1)) inputLayer[(midI+i+1)*28 + midJ+i+1]->connect(midLayer[connectMid]);
                }
            }
        }
        if (larger == false){//not modified yet, cause in MNIST case, larger always==true
            for (int i=0; i<midLayer.size(); i++) {
                connectNode = rand()%inputLayer.size();
                inputLayer[connectNode]->connect(midLayer[i]);
            }
        }
        
        //midlayer ---> outlayer
        for (int i=0; i<midLayer.size(); i++) {
            midLayer[i]->connect(outputLayer[i]);
        }
    }
    void randomConnect(int connectNode){
        bool isBreak = false;
        for (int i=0; i<inputLayer.size() && isBreak==false; i++) {
            for (int j=0; j<midLayer.size() && isBreak==false; j++) {
                int tmpR1 = rand()%inputLayer.size();
                int tmpR2 = rand()%midLayer.size();
                if (!isLoop(inputLayer[tmpR1], midLayer[tmpR2]) && abs(inputLayer[tmpR1]->delta)>320 && abs(midLayer[tmpR2]->delta)>320) {
                    inputLayer[tmpR1]->connect(midLayer[tmpR2]);
                    isBreak = true;
                }
            }
            isBreak = false;
            for (int j=0; j<outputLayer.size() && isBreak==false; j++) {
                int tmpR1 = rand()%inputLayer.size();
                int tmpR2 = rand()%outputLayer.size();
                if (!isLoop(inputLayer[tmpR1], outputLayer[tmpR2]) && abs(inputLayer[tmpR1]->delta)>320 && abs(outputLayer[tmpR2]->delta)>320){
                    inputLayer[tmpR1]->connect(outputLayer[tmpR2]);
                    isBreak = true;
                }
            }
        }
        
        isBreak = false;
        for (int i=0; i<midLayer.size() && isBreak==false; i++) {
            for(int j=0; j<outputLayer.size() && isBreak==false; j++){
                int tmpR1 = rand()%midLayer.size();
                int tmpR2 = rand()%outputLayer.size();
                if (!isLoop(midLayer[tmpR1], outputLayer[tmpR2]) && abs(midLayer[tmpR1]->delta)>320 && abs(outputLayer[tmpR2]->delta)>320) {
                    midLayer[tmpR1]->connect(outputLayer[tmpR2]);
                    isBreak = true;
                }
            }
        }
    }
private:
    int killThreshold = 5;
    double killThreshSum = 3.5;
    int nodeIDcount = 0;
    bool trained = false;
    vector<node*> inputLayer;
    vector<node*> midLayer;
    vector<node*> outputLayer;
    
    bool outputAllConnect(){//temporary definition
        for (int i=0; i<outputLayer.size(); i++) {
            if (outputLayer[i]->backwardConnection.size()<5) {
                return false;
            }
        }
        return true;
    }
    
    bool isLoop(node *x, node *y){
        if(x->ID < y->ID) return false;
        return true;
    }
    
    int dis(node*x, node*y){
        return abs((x->CAC.xMean-y->CAC.xMean) + (x->CAC.yMean-x->CAC.yMean));
    }
    
    vector<node*> nodesNearTo(node* x){
        vector<node*> rtn;
        for (int i=0; i<midLayer.size(); i++) {
            for (int j=0; j<rtn.size(); j++) {
                if (rtn.size()<4) {
                    rtn.push_back(midLayer[i]);
                    break;
                }
                else{
                    if (dis(midLayer[i], x) < dis(rtn[j], x)) {
                        rtn[j] = midLayer[i];
                    }
                }
            }
        }
        return rtn;
    }
    
};


#endif /* NEURALNET_hpp */
