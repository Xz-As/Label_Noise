# Introduction

This working dataset is numbered as follows:

### FashionMNIST0.5: 0

### FashionMNIST0.6: 1

### CIFAR-10: 2

A total of three neural networks have been implemented in this work and their names are as follows:

### CNN

### CNN+attention

### resnet18

### resnet20

Of these, resnet20 is a network structure specifically designed for the cifar dataset in place of the other resnet. Although all three network structures can be trained and inferred on any dataset, the authors used CNN, resnet18 on the other datasets and replaced resnet18 with resnet20 on the cifar dataset for scientific purposes.

In addition, to estimate the transfer matrix, two estimators are used in this work, a single anchor estimation method and a multi-anchor estimation method.

## Requirements

python>=3.6.6

torch>=1.9

torchvision>=0.7.0

numpy>=1.19.5

Datasets should be downloaded to ./data/

## Structure

```
-520322750_520316760
    |-data
    |-src
       |-dataset.py
       |-main.py
       |-experiment.py
       |-Noise_Adapt_Master
          |-algorithms_.py
          |-estimation.py
          |-utils.py
```

# Demo

Please redirect the run directory to the `filepath`.

```
cd 520322750_520316760
```

### To run CNN on all datasets, run.

```
$ python ./src/experiment.py --model_name CNN
```

### To run attention based CNN on all datasets, run.

```
$ python ./src/experiment.py --model_name CNN_attn --epoch 10 --epoch_fw 70
```

### To run with resnet18 on two MNIST datasets, run.

```
$ python ./src/experiment.py --model_name resnet18 --epoch 12 --epoch_fw 40
```

### To run resnet20 on CIFAR-10 dataset, run.

```
$ python ./src/experiment.py --model_name resnet20 ---epoch 15 -epoch_fw 60
```

### To run a model on MNIST0.5 datasets, please add parameter.

```
 --dataset 0
```

### To run a model on MNIST0.6 datasets, please add parameter.

```
 --dataset 1
```

### To run a model on CIFAR datasets, please add parameter.

```
 --dataset 2
```

### To run 10-times validation, please add parameter.

```
 --times 10
```

# Execution

Please redirect the run directory to the `filepath`.

```
cd 520322750_520316760
```

 Note: We strongly advise you to run the expirement SEPERATLY. Because each model has it's own best parameters and if all experiments were run simultaneously, all models would share one set of parameters, which would prevent some models from achieving the same performance shown in the paper. Also, the time of runing all models and datasets together will easily get out of control if some parameters are not small enough. 

## Basic commands

### To run experiments seperatly, please add some parameters on this basic command:

```
python ./src/main.py
```

### To run experiments simultaneously, please add some parameters on this basic command:

```
python ./src/experiment.py
```

## Parameters

### To run a specific dataset in a single experiment, run:

```
$ [command] --dataset <dataset number>
```

Example.

```
$ python ./src/main.py --dataset 0
```

The following parameters are available in either full experiments or single experiments.

### For experiments with a specific model, run:

```
[command] --model_name <model name>
```

Example.

```
python ./src/experiment.py --model_name resnet18
```

or

```
$ python ./src/main.py --model_name CNN
```

Specially, to experiment with all models, run:

```
$ python ./src/experiment.py --model_name all
```

#### To add a specific GPU to increase the training speed (cuda0 is used by default), run:

```
[command] --cuda cuda serie
```

Example.

```
$ python ./src/main.py --model_name resnet18 --dataset 0 --cuda 0
```

Specially, to use more than one GPUs, set the cuda series with number and comma

```
$ python ./src/main.py --model_name resnet18 --dataset 0 --cuda 0,1
```

To run the program on the CPU, set the cuda series to -

```
$ python ./src/main.py --model_name resnet18 --dataset 0 --cuda -
```

Other important parameters are used in the same way as above and are described as follows.

```
--times: int, the number of repetitions per experiment, default is 1, for a demo. If you want to run an entire 10-times resampling validation as shown in the paper, please set 10. It would take around 4~6 hours for CNN and around 20~30 hours for resnet in a single dataset under the parameters used in the paper. So, to validate the results in the paper, the authors recommend you to set a smaller value to save time; 1 is also welcome.

--flip_augment: bool, whether or not to add random image flipping as a data expansion method, default is True.

--lr: float, learning rate, the default is 0.1.

--lr_rd: float, learning rate, the default is 1. Please set 3 when running resnet.

--epoch: int the total epochs in the first training pocess of forward learning, default is 10. Please set 20 wen using resnet.

--epoch_fw: int, the total epochs in the second training process of forward learning, default is 30. Please set 60 when using resnet18 or using CNN on CIFAR, and set 120 when using resnet20.

--batch_size: int, batch size, default is 1024.

--num_workers: int, number of workers used to load dataloader, default is 8, please set samller if the pipe is broken or GPU is out of memory.

--multi_anchor: bool, whether to use multi-anchor method to estimate the transition matrix, default is True.
```

# Results

The path to save the results of each experiment is: 

```
./results/<dataset name>/<model name>/<number of runs>/
```

Example.

```
./results/CIFAR/CNN/1/
```

Under each `<model name>` file, the mean and standard deviation of the top-1 ACC after 10-times validations are saved in `Final_ACC.csv`.

Note: The `<number of run>` will always start from 0. And the new experiment will rewrite the old ones if the dataset and model are the same.

