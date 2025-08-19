#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <time.h>


#define numOfInputs 3
#define numOfOutputs 2
#define numOfHiddenNurons 3
#define numOfTrainingSets 8

#define numOfWeightsIn 3
#define numOFWeightsHide1 5
#define numOfWeightsOut 5

double sigmoid(double x){
    return 1 / (1 + exp( -x ));
}

double dSigmoid(double x){
    return x * (1 - x );
}

double dSigmoid1(double x){
    return exp( -x ) / pow(1 + exp( -x ), 2);
}

double init_weight(){
    return ((double)rand() / ((double)RAND_MAX));
}



int main(void){


    srand(time(NULL));

    // learning rate 
    const double lr = 0.1f;

    // The Network parts
    double InputLayer[numOfInputs];
    double OutputLayer[numOfOutputs];
    double HiddenLayer1[numOfHiddenNurons];
    double HiddenLayer2[numOfHiddenNurons];

    // C A B
    const double InputTrain[numOfTrainingSets][numOfInputs] = {{1,1,0},{1,0,1},{1,0,0},{1,1,1},{0,1,0},{0,0,1},{0,0,0},{0,1,1}};
    const double OutputTrain[numOfTrainingSets][numOfOutputs] = {{1,0},{1,0},{1,0},{0,1},{0,1},{0,1},{0,1},{1,0}};
    // Weight Matrices
    double InWeightMatrix[numOfInputs][numOfHiddenNurons];
    double HideWeightMatrix[numOfHiddenNurons][numOfHiddenNurons];
    double OutWeightMatrix[numOfHiddenNurons][numOfOutputs];


    // bias arrays

    double HiddenLayer1Bias[numOfHiddenNurons];
    double HiddenLayer2Bias[numOfHiddenNurons];
    double OutputLayerBias[numOfOutputs];

    // error arrays
    double ErrorHidden1[numOfHiddenNurons];
    double ErrorHidden2[numOfHiddenNurons];
    double ErrorOutput[numOfOutputs];

    // random values in weight matrices

    for(int i = 0; i < numOfInputs; i++){
        for(int j = 0; j < numOfHiddenNurons; j++){
            InWeightMatrix[i][j] = init_weight();
        }
    }

    
    for(int i = 0; i < numOfHiddenNurons; i++){
        for(int j = 0; j < numOfHiddenNurons; j++){
            HideWeightMatrix[i][j] = init_weight();
        }
    }

    
    for(int i = 0; i < numOfHiddenNurons; i++){
        for(int j = 0; j < numOfOutputs; j++){
            OutWeightMatrix[i][j] = init_weight();
        }
    }



    // random bias

    for(int i = 0; i < numOfHiddenNurons; i++){
        HiddenLayer1Bias[i] = init_weight();
    }

    
    for(int i = 0; i < numOfHiddenNurons; i++){
        HiddenLayer2Bias[i] = init_weight();
    }

    
    for(int i = 0; i < numOfOutputs; i++){
        OutputLayerBias[i] = init_weight();
    }

    // int learningOrder[numOfTrainingSets] = {0,3,5,1,4,7,2,6};

    int numOfEpochs = 10000;

    // training the neural network

    for(int epoch = 0; epoch < numOfEpochs; epoch++){

        for(int x = 0; x < numOfTrainingSets; x++){

            // setting the inputs
            for(int t = 0; t < numOfInputs; t++){
                InputLayer[t] = InputTrain[x][t];
            }

            
            // forward propagation

            // calculating the first hiding layer


            for(int i = 0; i < numOfHiddenNurons; i++){
                HiddenLayer1[i] = 0.0f;
                for(int j = 0; j < numOfInputs; j++){
                    HiddenLayer1[i] += InputLayer[j] * InWeightMatrix[j][i];
                }
                HiddenLayer1[i] = sigmoid(HiddenLayer1[i] + HiddenLayer1Bias[i]);
            }


            // calculating the the seconde hiding layer
            
            for(int i = 0; i < numOfHiddenNurons; i++){
                HiddenLayer2[i] = 0.0f;
                for(int j = 0; j < numOfHiddenNurons; j++){
                    HiddenLayer2[i] += HiddenLayer1[j] * HideWeightMatrix[j][i];
                }
                HiddenLayer2[i] = sigmoid(HiddenLayer2[i] + HiddenLayer2Bias[i]);
            }


            // calculating the output layer

            
            for(int i = 0; i < numOfOutputs; i++){
                OutputLayer[i] = 0.0f;
                for(int j = 0; j < numOfHiddenNurons; j++){
                    OutputLayer[i] += HiddenLayer2[j] * OutWeightMatrix[j][i];
                }
                OutputLayer[i] = sigmoid(OutputLayer[i] + OutputLayerBias[i]);
            }
        
            printf("Input:\n \tC: %g\n \tA: %g\n \tB: %g\n  Output: \n \t0: %g\n \t1: %g\n    Expected: \n \t0: %g\n \t1: %g\n", InputLayer[0], InputLayer[1], InputLayer[2], OutputLayer[0], OutputLayer[1], OutputTrain[x][0], OutputTrain[x][1]);

        
            // calc the error
            
            // output layer error

            for(int i = 0; i < numOfOutputs; i++){
                ErrorOutput[i] = (OutputTrain[x][i] - OutputLayer[i]) * dSigmoid(OutputLayer[i]);
            }

            // hidden layer 2 error

            for(int i = 0; i < numOfHiddenNurons; i++){
                double error = 0.0f;
                for(int j = 0 ; j < numOfOutputs; j++){
                    error += ErrorOutput[j] * OutWeightMatrix[i][j]; 
                }
                ErrorHidden2[i] = error * dSigmoid(HiddenLayer2[i]);
            }

            // hidden layer 1 error

            for(int i = 0; i < numOfHiddenNurons; i++){
                double error = 0.0f;
                for(int j = 0 ; j < numOfHiddenNurons; j++){
                    error += ErrorHidden2[j] * HideWeightMatrix[i][j]; 
                }
                ErrorHidden1[i] = error * dSigmoid(HiddenLayer1[i]);
            }


            //updating the weights

            // output weights

            for(int i = 0; i < numOfOutputs; i++){
                // update bias
                OutputLayerBias[i] += ErrorOutput[i] * lr;
                for(int j = 0; j < numOfHiddenNurons; j++){
                    OutWeightMatrix[j][i] += HiddenLayer2[j] * ErrorOutput[i] * lr; 
                }
            }


            // hidden weights

            for(int i = 0; i < numOfHiddenNurons; i++){
                // update bias
                HiddenLayer2Bias[i] += ErrorHidden2[i] * lr;
                for(int j = 0; j < numOfHiddenNurons; j++){
                    HideWeightMatrix[j][i] += HiddenLayer1[j] * ErrorHidden2[i] * lr; 
                }
            }
            

            // input weights

            for(int i = 0; i < numOfHiddenNurons; i++){
                // update bias
                HiddenLayer1Bias[i] += ErrorHidden1[i] * lr;
                for(int j = 0; j < numOfInputs; j++){
                    InWeightMatrix[j][i] += InputLayer[j] * ErrorHidden1[i] * lr; 
                }
            }


        
        }




    }

    
    printf("finel input weights:\n");
    for(int i = 0; i < numOfInputs; i++){
        for(int j = 0; j < numOfHiddenNurons; j++){
            printf("\t[%g]\t", InWeightMatrix[i][j]);
        }
        printf("\n");
    }



    printf("final hidden layer 1 bias:\n");
    for(int i = 0; i < numOfHiddenNurons; i++){
        printf("\t[%g]\n", HiddenLayer1Bias[i]);
    }

    printf("finel hidden layer 1 weights:\n");
    for(int i = 0; i < numOfHiddenNurons; i++){
        for(int j = 0; j < numOfHiddenNurons; j++){
            printf("\t[%g]\t", OutWeightMatrix[i][j]);
        }
        printf("\n");
    }


    printf("final hidden layer 2 bias:\n");
    for(int i = 0; i < numOfHiddenNurons; i++){
        printf("\t[%g]\n", HiddenLayer2Bias[i]);
    }

    printf("finel hidden layer 2 weights:\n");
    for(int i = 0; i < numOfHiddenNurons; i++){
        for(int j = 0; j < numOfOutputs; j++){
            printf("\t[%g]\t", HideWeightMatrix[i][j]);
        }
        printf("\n");
    }


    
    printf("final output bias:\n");
    for(int i = 0; i < numOfOutputs; i++){
        printf("\t[%g]\n", OutputLayerBias[i]);
    }



}