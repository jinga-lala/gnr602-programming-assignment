# GNR 602- Programming Assignment
### Yash Jain (170050055), Rishav Arjun (170100051)

## Problem Statement:
Implement a single layer perceptron classifier (input layer + output layer without any
hidden layer) with a polynomial function for non-linear transformation of the input. Compare
this result with the result when no non-linear transformation of the input is done.

## Datasets
- OR dataset: Test the basic perceptron implementation
- Toy dataset: Test the implementation of nonlinear transformation and demonstrate its usefulness
- SONAR dataset: Real world dataset from UCI Machine learning repository 
  [http://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)]
  
## Running commands 

```
python main.py --data or --epoch 20 --transform 0 --learning_rate 0.1

python main.py --data toy --epoch 30 --transform 0 --learning_rate 0.001
python main.py --data toy --epoch 30 --transform 1 --learning_rate 0.001

python main.py --data sonar --epoch 100 --transform 0 --learning_rate 0.001
python main.py --data sonar --epoch 100 --transform 1 --learning_rate 0.001

python main.py --data image
```
**By selecting "image" option you will be prompted to select train image, train label, test image, test label and output directory to store the predicted image segmented**
Read the argparse help messages to know tune the hyperparameters

