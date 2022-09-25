#!/bin/bash

gpuid=0
if [ $# -eq 0 ]; then
  echo "Please select one or more datasets: perMNIST rotMNIST CIFAR100 miniImageNet"
  exit
fi
for exp in "$@"; do
  echo "Perform Experiment ER-PFCL on $exp"
  if [ "$exp" = "perMNIST" ]; then
    python main.py --dataset perMNIST --scenario domain --tasks 20 --scheme ER_PFCL --optim DCD --lr 0.01 --decay 0.01 --momentum 0.7 --memory 1 --bs 10 --epochs 1 --omega 20 --lambd 80 --beta 0.01 --eta 0 --recall --theta 20 --delta 0.1 --mode sphere --gpuid $gpuid --name er-pfcl/Domain-IL/perMNIST --seed 1234
    python main.py --dataset perMNIST --scenario domain --tasks 20 --scheme ER_PFCL --optim DCD --lr 0.01 --decay 0.01 --momentum 0.7 --memory 1 --bs 10 --epochs 1 --omega 20 --lambd 80 --beta 0.01 --eta 0 --recall --theta 20 --delta 0.1 --mode sphere --gpuid $gpuid --name er-pfcl/Domain-IL/perMNIST --seed 4567
    python main.py --dataset perMNIST --scenario domain --tasks 20 --scheme ER_PFCL --optim DCD --lr 0.01 --decay 0.01 --momentum 0.7 --memory 1 --bs 10 --epochs 1 --omega 20 --lambd 80 --beta 0.01 --eta 0 --recall --theta 20 --delta 0.1 --mode sphere --gpuid $gpuid --name er-pfcl/Domain-IL/perMNIST --seed 7891
    python main.py --dataset perMNIST --scenario domain --tasks 20 --scheme ER_PFCL --optim DCD --lr 0.01 --decay 0.01 --momentum 0.7 --memory 1 --bs 10 --epochs 1 --omega 20 --lambd 80 --beta 0.01 --eta 0 --recall --theta 20 --delta 0.1 --mode sphere --gpuid $gpuid --name er-pfcl/Domain-IL/perMNIST --seed 7295
    python main.py --dataset perMNIST --scenario domain --tasks 20 --scheme ER_PFCL --optim DCD --lr 0.01 --decay 0.01 --momentum 0.7 --memory 1 --bs 10 --epochs 1 --omega 20 --lambd 80 --beta 0.01 --eta 0 --recall --theta 20 --delta 0.1 --mode sphere --gpuid $gpuid --name er-pfcl/Domain-IL/perMNIST --seed 5234
    python main.py --name er-pfcl/Domain-IL/perMNIST --opt summary

  elif [ "$exp" = "rotMNIST" ]; then
    python main.py --dataset rotMNIST --scenario domain --tasks 20 --scheme ER_PFCL --optim DCD --lr 0.01 --decay 5e-5 --momentum 0.7 --memory 1 --bs 10 --epochs 1 --omega 20 --lambd 10 --beta 0.01 --eta 0 --recall --theta 20 --delta 0.1 --mode multi-sphere --gpuid $gpuid --name er-pfcl/Domain-IL/rotMNIST --seed 1234
    python main.py --dataset rotMNIST --scenario domain --tasks 20 --scheme ER_PFCL --optim DCD --lr 0.01 --decay 5e-5 --momentum 0.7 --memory 1 --bs 10 --epochs 1 --omega 20 --lambd 10 --beta 0.01 --eta 0 --recall --theta 20 --delta 0.1 --mode multi-sphere --gpuid $gpuid --name er-pfcl/Domain-IL/rotMNIST --seed 4567
    python main.py --dataset rotMNIST --scenario domain --tasks 20 --scheme ER_PFCL --optim DCD --lr 0.01 --decay 5e-5 --momentum 0.7 --memory 1 --bs 10 --epochs 1 --omega 20 --lambd 10 --beta 0.01 --eta 0 --recall --theta 20 --delta 0.1 --mode multi-sphere --gpuid $gpuid --name er-pfcl/Domain-IL/rotMNIST --seed 7891
    python main.py --dataset rotMNIST --scenario domain --tasks 20 --scheme ER_PFCL --optim DCD --lr 0.01 --decay 5e-5 --momentum 0.7 --memory 1 --bs 10 --epochs 1 --omega 20 --lambd 10 --beta 0.01 --eta 0 --recall --theta 20 --delta 0.1 --mode multi-sphere --gpuid $gpuid --name er-pfcl/Domain-IL/rotMNIST --seed 7295
    python main.py --dataset rotMNIST --scenario domain --tasks 20 --scheme ER_PFCL --optim DCD --lr 0.01 --decay 5e-5 --momentum 0.7 --memory 1 --bs 10 --epochs 1 --omega 20 --lambd 10 --beta 0.01 --eta 0 --recall --theta 20 --delta 0.1 --mode multi-sphere --gpuid $gpuid --name er-pfcl/Domain-IL/rotMNIST --seed 5234
    python main.py --name er-pfcl/Domain-IL/rotMNIST --opt summary

  elif [ "$exp" = "CIFAR100" ]; then
    python main.py --dataset CIFAR100 --scenario task --tasks 20 --scheme ER_PFCL --optim DCD --lr 0.1 --decay 1e-4 --momentum 0.7 --memory 1 --bs 10 --epochs 1 --omega 20 --lambd 20 --beta 0.01 --eta 0.3 --recall --theta 20 --delta 0.1 --mode sphere --gpuid $gpuid --name er-pfcl/Task-IL/CIFAR100 --seed 1234
    python main.py --dataset CIFAR100 --scenario task --tasks 20 --scheme ER_PFCL --optim DCD --lr 0.1 --decay 1e-4 --momentum 0.7 --memory 1 --bs 10 --epochs 1 --omega 20 --lambd 20 --beta 0.01 --eta 0.3 --recall --theta 20 --delta 0.1 --mode sphere --gpuid $gpuid --name er-pfcl/Task-IL/CIFAR100 --seed 4567
    python main.py --dataset CIFAR100 --scenario task --tasks 20 --scheme ER_PFCL --optim DCD --lr 0.1 --decay 1e-4 --momentum 0.7 --memory 1 --bs 10 --epochs 1 --omega 20 --lambd 20 --beta 0.01 --eta 0.3 --recall --theta 20 --delta 0.1 --mode sphere --gpuid $gpuid --name er-pfcl/Task-IL/CIFAR100 --seed 7891
    python main.py --dataset CIFAR100 --scenario task --tasks 20 --scheme ER_PFCL --optim DCD --lr 0.1 --decay 1e-4 --momentum 0.7 --memory 1 --bs 10 --epochs 1 --omega 20 --lambd 20 --beta 0.01 --eta 0.3 --recall --theta 20 --delta 0.1 --mode sphere --gpuid $gpuid --name er-pfcl/Task-IL/CIFAR100 --seed 7295
    python main.py --dataset CIFAR100 --scenario task --tasks 20 --scheme ER_PFCL --optim DCD --lr 0.1 --decay 1e-4 --momentum 0.7 --memory 1 --bs 10 --epochs 1 --omega 20 --lambd 20 --beta 0.01 --eta 0.3 --recall --theta 20 --delta 0.1 --mode sphere --gpuid $gpuid --name er-pfcl/Task-IL/CIFAR100 --seed 5234
    python main.py --name er-pfcl/Task-IL/CIFAR100 --opt summary

  elif [ "$exp" = "miniImageNet" ]; then
    python main.py --dataset miniImageNet --scenario task --tasks 20 --scheme ER_PFCL --optim DCD --lr 0.1 --decay 1e-5 --momentum 0.7 --memory 1 --bs 10 --epochs 1 --omega 20 --lambd 160 --beta 0.01 --eta 0 --recall --theta 40 --delta 0.1 --mode sphere --gpuid $gpuid --name er-pfcl/Task-IL/miniImageNet --seed 1234
    python main.py --dataset miniImageNet --scenario task --tasks 20 --scheme ER_PFCL --optim DCD --lr 0.1 --decay 1e-5 --momentum 0.7 --memory 1 --bs 10 --epochs 1 --omega 20 --lambd 160 --beta 0.01 --eta 0 --recall --theta 40 --delta 0.1 --mode sphere --gpuid $gpuid --name er-pfcl/Task-IL/miniImageNet --seed 4567
    python main.py --dataset miniImageNet --scenario task --tasks 20 --scheme ER_PFCL --optim DCD --lr 0.1 --decay 1e-5 --momentum 0.7 --memory 1 --bs 10 --epochs 1 --omega 20 --lambd 160 --beta 0.01 --eta 0 --recall --theta 40 --delta 0.1 --mode sphere --gpuid $gpuid --name er-pfcl/Task-IL/miniImageNet --seed 7891
    python main.py --dataset miniImageNet --scenario task --tasks 20 --scheme ER_PFCL --optim DCD --lr 0.1 --decay 1e-5 --momentum 0.7 --memory 1 --bs 10 --epochs 1 --omega 20 --lambd 160 --beta 0.01 --eta 0 --recall --theta 40 --delta 0.1 --mode sphere --gpuid $gpuid --name er-pfcl/Task-IL/miniImageNet --seed 7295
    python main.py --dataset miniImageNet --scenario task --tasks 20 --scheme ER_PFCL --optim DCD --lr 0.1 --decay 1e-5 --momentum 0.7 --memory 1 --bs 10 --epochs 1 --omega 20 --lambd 160 --beta 0.01 --eta 0 --recall --theta 40 --delta 0.1 --mode sphere --gpuid $gpuid --name er-pfcl/Task-IL/miniImageNet --seed 5234
    python main.py --name er-pfcl/Task-IL/miniImageNet --opt summary

  else
    echo "Error Dataset: $exp"
  fi
done
