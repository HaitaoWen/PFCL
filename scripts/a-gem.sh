#!/bin/bash

gpuid=0
if [ $# -eq 0 ]; then
  echo "Please select one or more datasets: perMNIST rotMNIST CIFAR100 miniImageNet"
  exit
fi
for exp in "$@"; do
  echo "Perform Experiment A-GEM on $exp"
  if [ "$exp" = "perMNIST" ]; then
    python main.py --dataset perMNIST --scenario domain --tasks 20 --scheme GEM --lr 0.01 --decay 0 --momentum 0.7 --memory 1 --mcat --bs 10 --mbs 10 --epochs 1 --gpuid $gpuid --name a-gem/Domain-IL/perMNIST --opt A-GEM --seed 1234
    python main.py --dataset perMNIST --scenario domain --tasks 20 --scheme GEM --lr 0.01 --decay 0 --momentum 0.7 --memory 1 --mcat --bs 10 --mbs 10 --epochs 1 --gpuid $gpuid --name a-gem/Domain-IL/perMNIST --opt A-GEM --seed 4567
    python main.py --dataset perMNIST --scenario domain --tasks 20 --scheme GEM --lr 0.01 --decay 0 --momentum 0.7 --memory 1 --mcat --bs 10 --mbs 10 --epochs 1 --gpuid $gpuid --name a-gem/Domain-IL/perMNIST --opt A-GEM --seed 7891
    python main.py --dataset perMNIST --scenario domain --tasks 20 --scheme GEM --lr 0.01 --decay 0 --momentum 0.7 --memory 1 --mcat --bs 10 --mbs 10 --epochs 1 --gpuid $gpuid --name a-gem/Domain-IL/perMNIST --opt A-GEM --seed 7295
    python main.py --dataset perMNIST --scenario domain --tasks 20 --scheme GEM --lr 0.01 --decay 0 --momentum 0.7 --memory 1 --mcat --bs 10 --mbs 10 --epochs 1 --gpuid $gpuid --name a-gem/Domain-IL/perMNIST --opt A-GEM --seed 5234
    python main.py --name a-gem/Domain-IL/perMNIST --opt summary

  elif [ "$exp" = "rotMNIST" ]; then
    python main.py --dataset rotMNIST --scenario domain --tasks 20 --scheme GEM --lr 0.01 --decay 0 --momentum 0.7 --memory 1 --mcat --bs 10 --mbs 10 --epochs 1 --gpuid $gpuid --name a-gem/Domain-IL/rotMNIST --opt A-GEM --seed 1234
    python main.py --dataset rotMNIST --scenario domain --tasks 20 --scheme GEM --lr 0.01 --decay 0 --momentum 0.7 --memory 1 --mcat --bs 10 --mbs 10 --epochs 1 --gpuid $gpuid --name a-gem/Domain-IL/rotMNIST --opt A-GEM --seed 4567
    python main.py --dataset rotMNIST --scenario domain --tasks 20 --scheme GEM --lr 0.01 --decay 0 --momentum 0.7 --memory 1 --mcat --bs 10 --mbs 10 --epochs 1 --gpuid $gpuid --name a-gem/Domain-IL/rotMNIST --opt A-GEM --seed 7891
    python main.py --dataset rotMNIST --scenario domain --tasks 20 --scheme GEM --lr 0.01 --decay 0 --momentum 0.7 --memory 1 --mcat --bs 10 --mbs 10 --epochs 1 --gpuid $gpuid --name a-gem/Domain-IL/rotMNIST --opt A-GEM --seed 7295
    python main.py --dataset rotMNIST --scenario domain --tasks 20 --scheme GEM --lr 0.01 --decay 0 --momentum 0.7 --memory 1 --mcat --bs 10 --mbs 10 --epochs 1 --gpuid $gpuid --name a-gem/Domain-IL/rotMNIST --opt A-GEM --seed 5234
    python main.py --name a-gem/Domain-IL/rotMNIST --opt summary

  elif [ "$exp" = "CIFAR100" ]; then
    python main.py --dataset CIFAR100 --scenario task --tasks 20 --scheme GEM --lr 0.1 --decay 0 --momentum 0.7 --memory 1 --mcat --bs 10 --mbs 10 --epochs 1 --gpuid $gpuid --name a-gem/Task-IL/CIFAR100 --opt A-GEM --seed 1234
    python main.py --dataset CIFAR100 --scenario task --tasks 20 --scheme GEM --lr 0.1 --decay 0 --momentum 0.7 --memory 1 --mcat --bs 10 --mbs 10 --epochs 1 --gpuid $gpuid --name a-gem/Task-IL/CIFAR100 --opt A-GEM --seed 4567
    python main.py --dataset CIFAR100 --scenario task --tasks 20 --scheme GEM --lr 0.1 --decay 0 --momentum 0.7 --memory 1 --mcat --bs 10 --mbs 10 --epochs 1 --gpuid $gpuid --name a-gem/Task-IL/CIFAR100 --opt A-GEM --seed 7891
    python main.py --dataset CIFAR100 --scenario task --tasks 20 --scheme GEM --lr 0.1 --decay 0 --momentum 0.7 --memory 1 --mcat --bs 10 --mbs 10 --epochs 1 --gpuid $gpuid --name a-gem/Task-IL/CIFAR100 --opt A-GEM --seed 7295
    python main.py --dataset CIFAR100 --scenario task --tasks 20 --scheme GEM --lr 0.1 --decay 0 --momentum 0.7 --memory 1 --mcat --bs 10 --mbs 10 --epochs 1 --gpuid $gpuid --name a-gem/Task-IL/CIFAR100 --opt A-GEM --seed 5234
    python main.py --name a-gem/Task-IL/CIFAR100 --opt summary

  elif [ "$exp" = "miniImageNet" ]; then
    python main.py --dataset miniImageNet --scenario task --tasks 20 --scheme GEM --lr 0.1 --decay 0 --momentum 0.7 --memory 1 --mcat --bs 10 --mbs 10 --epochs 1 --gpuid $gpuid --name a-gem/Task-IL/miniImageNet --opt A-GEM --seed 1234
    python main.py --dataset miniImageNet --scenario task --tasks 20 --scheme GEM --lr 0.1 --decay 0 --momentum 0.7 --memory 1 --mcat --bs 10 --mbs 10 --epochs 1 --gpuid $gpuid --name a-gem/Task-IL/miniImageNet --opt A-GEM --seed 4567
    python main.py --dataset miniImageNet --scenario task --tasks 20 --scheme GEM --lr 0.1 --decay 0 --momentum 0.7 --memory 1 --mcat --bs 10 --mbs 10 --epochs 1 --gpuid $gpuid --name a-gem/Task-IL/miniImageNet --opt A-GEM --seed 7891
    python main.py --dataset miniImageNet --scenario task --tasks 20 --scheme GEM --lr 0.1 --decay 0 --momentum 0.7 --memory 1 --mcat --bs 10 --mbs 10 --epochs 1 --gpuid $gpuid --name a-gem/Task-IL/miniImageNet --opt A-GEM --seed 7295
    python main.py --dataset miniImageNet --scenario task --tasks 20 --scheme GEM --lr 0.1 --decay 0 --momentum 0.7 --memory 1 --mcat --bs 10 --mbs 10 --epochs 1 --gpuid $gpuid --name a-gem/Task-IL/miniImageNet --opt A-GEM --seed 5234
    python main.py --name a-gem/Task-IL/miniImageNet --opt summary

  else
    echo "Error Dataset: $exp"
  fi
done
