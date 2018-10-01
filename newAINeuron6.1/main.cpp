














#include "NEURALNET3.hpp"
#include "mnist.h"
int const dataSize = 3200;
double beta  = 0.9;
int midLayerNode = 16;
int inputDim = 28*28;
int outputDim = 10;
int evalInterval = 0;
extern double T;
extern double ALPHA;
extern int const trainingTime;


void testAccuracy(NeuralNet &nn ,vector<vector<double>> &x, vector<vector<double>> &y){
    float correct = 0;
    vector<double> rslt;
    int max;
    int maxInd;
    for (int i=0; i<200; i++) {
        int rnd = rand()%3200 + 3200;
        rslt.clear();
        rslt = nn.inference(x[rnd], false);
        max = -100;
        maxInd = 0;
        for (int j=0; j<10; j++) {
            if (max <= rslt[j]) {
                max = rslt[j];
                maxInd = j;
            }
        }
        if (y[rnd][maxInd] >= 0.99) {
            correct += 1;
        }
    }
    printf("Accuracy: %.2f%%\n\n", correct*100/200.0);
}

double calculateSumOfLoss(int size, NeuralNet &nn, vector<vector<double>> &x, vector<vector<double>> &y, int offset){
    double sumOfLoss = 0;
    for (int j=0; j<size; j++){
        int tmp = rand()%dataSize + offset;
        sumOfLoss = sumOfLoss + nn.calculateLoss(x[tmp], y[tmp]);
    }
    return sumOfLoss;
}

void printSeveralTrainingCase(int size, NeuralNet &nn , vector<vector<double>> &x, vector<vector<double>> &y) {
    vector<double> tmpVect;
    for (int j=0; j<size; j = j+1) {
        int rnd = rand()%dataSize;
        tmpVect.clear();
        tmpVect = nn.inference(x[rnd], false);//false --> not training Phase
        for (int k=0; k<10; k++) {
            printf("x: %lf  infr: %.3lf\n", y[rnd][k], tmpVect[k]);
        }
        printf("\n");
    }
}
void trainNeuralNet(NeuralNet &nn, int size, vector<vector<double>>&x, vector<vector<double>>&y){
    int rnd;
    for (int j=0; j<size; j++) {
        rnd = rand()%dataSize;
        nn.train(x[rnd], y[rnd]);
        T++;
    }
}
int main() {
    //seed random
    srand (time(NULL));

    // call to store mnist in array
    //test_image, test_label, train_image, train_label become available
    load_mnist();
    
    //instantiation of data
    vector<vector<double>> x;
    vector<vector<double>> y;
    x.resize(2*dataSize);
    y.resize(2*dataSize);
    
    //initializing data
    for (int i=0; i<2*dataSize; i++) {
        double sum=0;
        double avg=0;
        for (int j=0; j<28*28; j++) {
            x[i].push_back(train_image[i][j]);
            sum += (train_image[i][j]);
        }
        avg = avg/inputDim;
        for (int j=0; j<inputDim; j++) {
            x[i][j] = x[i][j] - avg;
        }
        for (int j=0; j<10; j++) {
            if (j==train_label[i]) y[i].push_back(1.0);
            else y[i].push_back(0.0);
        }
    }
    
    //instantiation of an Neural net
    NeuralNet nn(inputDim, outputDim, midLayerNode); //(inputNode, outputNode, nodesInMiddle)
    printf("Initialized\n");
    double sumOfLoss = 0;
    double testSumOfLoss = 0;
    double preLoss = 100000000000;
    int Q = 20;
    for (int i=0; i<trainingTime; i++) {
        
        sumOfLoss = calculateSumOfLoss(50, nn, x, y, 0);//offset = 0 -->training data
        if (i%Q==0) {
            evalInterval += 5;
            Q = 2*Q;
        }
        if (i%evalInterval==0) {//Q%evalInterval==0
            if (preLoss*0.96<sumOfLoss && i<trainingTime) {
                printSeveralTrainingCase(10, nn, x, y);
                testAccuracy(nn, x, y);
                printf("NodeCount: %d\n", NodeCount);
                for (int j=0; j<8; j++) {
                    int tmp = rand()%dataSize;
                    nn.addNode(x[tmp], y[tmp]);
                }
                nn.clearMomentum();
            }
            //else{
            //    nn.Perturb();
            //}
            
            preLoss = sumOfLoss;
        }
        printf("Loss: %lf\n", sumOfLoss/(double)50);
        testSumOfLoss = calculateSumOfLoss(50, nn, x, y, 3200);//offset = 6400 -->out of training data
        printf("Test Loss %lf\n\n", testSumOfLoss/50.0);
        trainNeuralNet(nn, 800, x, y);
    }
    printSeveralTrainingCase(10, nn, x, y);
    return 0;
}
