#!/bin/bash

gpuid=0
if [ $# -eq 0 ]; then
  echo "Please select one or more datasets: perMNIST rotMNIST CIFAR100 miniImageNet"
  exit
fi
for exp in "$@"; do
  echo "Perform Experiment MultiTask on $exp"
  if [ "$exp" = "perMNIST" ]; then
    python main.py --dataset perMNIST --scenario domain --tasks 20 --scheme MultiTask --lr 0.01 --decay 0 --momentum 0.7 --bs 10 --epochs 1 --gpuid $gpuid --name multitask/Domain-IL/perMNIST --seed 1234
    python main.py --dataset perMNIST --scenario domain --tasks 20 --scheme MultiTask --lr 0.01 --decay 0 --momentum 0.7 --bs 10 --epochs 1 --gpuid $gpuid --name multitask/Domain-IL/perMNIST --seed 4567
    python main.py --dataset perMNIST --scenario domain --tasks 20 --scheme MultiTask --lr 0.01 --decay 0 --momentum 0.7 --bs 10 --epochs 1 --gpuid $gpuid --name multitask/Domain-IL/perMNIST --seed 7891
    python main.py --dataset perMNIST --scenario domain --tasks 20 --scheme MultiTask --lr 0.01 --decay 0 --momentum 0.7 --bs 10 --epochs 1 --gpuid $gpuid --name multitask/Domain-IL/perMNIST --seed 7295
    python main.py --dataset perMNIST --scenario domain --tasks 20 --scheme MultiTask --lr 0.01 --decay 0 --momentum 0.7 --bs 10 --epochs 1 --gpuid $gpuid --name multitask/Domain-IL/perMNIST --seed 5234
    python main.py --name multitask/Domain-IL/perMNIST --opt summary

  elif [ "$exp" = "rotMNIST" ]; then
    python main.py --dataset rotMNIST --scenario domain --tasks 20 --scheme MultiTask --lr 0.01 --decay 0 --momentum 0.7 --bs 10 --epochs 1 --gpuid $gpuid --name multitask/Domain-IL/rotMNIST --seed 1234
    python main.py --dataset rotMNIST --scenario domain --tasks 20 --scheme MultiTask --lr 0.01 --decay 0 --momentum 0.7 --bs 10 --epochs 1 --gpuid $gpuid --name multitask/Domain-IL/rotMNIST --seed 4567
    python main.py --dataset rotMNIST --scenario domain --tasks 20 --scheme MultiTask --lr 0.01 --decay 0 --momentum 0.7 --bs 10 --epochs 1 --gpuid $gpuid --name multitask/Domain-IL/rotMNIST --seed 7891
    python main.py --dataset rotMNIST --scenario domain --tasks 20 --scheme MultiTask --lr 0.01 --decay 0 --momentum 0.7 --bs 10 --epochs 1 --gpuid $gpuid --name multitask/Domain-IL/rotMNIST --seed 7295
    python main.py --dataset rotMNIST --scenario domain --tasks 20 --scheme MultiTask --lr 0.01 --decay 0 --momentum 0.7 --bs 10 --epochs 1 --gpuid $gpuid --name multitask/Domain-IL/rotMNIST --seed 5234
    python main.py --name multitask/Domain-IL/rotMNIST --opt summary

  elif [ "$exp" = "CIFAR100" ]; then
    python main.py --dataset CIFAR100 --scenario task --tasks 20 --scheme MultiTask --lr 0.1 --decay 1e-5 --momentum 0.7 --bs 10 --epochs 1 --gpuid $gpuid --name multitask/Task-IL/CIFAR100 --seed 1234
    python main.py --dataset CIFAR100 --scenario task --tasks 20 --scheme MultiTask --lr 0.1 --decay 1e-5 --momentum 0.7 --bs 10 --epochs 1 --gpuid $gpuid --name multitask/Task-IL/CIFAR100 --seed 4567
    python main.py --dataset CIFAR100 --scenario task --tasks 20 --scheme MultiTask --lr 0.1 --decay 1e-5 --momentum 0.7 --bs 10 --epochs 1 --gpuid $gpuid --name multitask/Task-IL/CIFAR100 --seed 7891
    python main.py --dataset CIFAR100 --scenario task --tasks 20 --scheme MultiTask --lr 0.1 --decay 1e-5 --momentum 0.7 --bs 10 --epochs 1 --gpuid $gpuid --name multitask/Task-IL/CIFAR100 --seed 7295
    python main.py --dataset CIFAR100 --scenario task --tasks 20 --scheme MultiTask --lr 0.1 --decay 1e-5 --momentum 0.7 --bs 10 --epochs 1 --gpuid $gpuid --name multitask/Task-IL/CIFAR100 --seed 5234
    python main.py --name multitask/Task-IL/CIFAR100 --opt summary

  elif [ "$exp" = "miniImageNet" ]; then
    python main.py --dataset miniImageNet --scenario task --tasks 20 --scheme MultiTask --lr 0.01 --decay 0 --momentum 0.7 --bs 10 --epochs 1 --gpuid $gpuid --name multitask/Task-IL/miniImageNet --seed 1234
    python main.py --dataset miniImageNet --scenario task --tasks 20 --scheme MultiTask --lr 0.01 --decay 0 --momentum 0.7 --bs 10 --epochs 1 --gpuid $gpuid --name multitask/Task-IL/miniImageNet --seed 4567
    python main.py --dataset miniImageNet --scenario task --tasks 20 --scheme MultiTask --lr 0.01 --decay 0 --momentum 0.7 --bs 10 --epochs 1 --gpuid $gpuid --name multitask/Task-IL/miniImageNet --seed 7891
    python main.py --dataset miniImageNet --scenario task --tasks 20 --scheme MultiTask --lr 0.01 --decay 0 --momentum 0.7 --bs 10 --epochs 1 --gpuid $gpuid --name multitask/Task-IL/miniImageNet --seed 7295
    python main.py --dataset miniImageNet --scenario task --tasks 20 --scheme MultiTask --lr 0.01 --decay 0 --momentum 0.7 --bs 10 --epochs 1 --gpuid $gpuid --name multitask/Task-IL/miniImageNet --seed 5234
    python main.py --name multitask/Task-IL/miniImageNet --opt summary

  else
    echo "Error Dataset: $exp"
  fi
done
