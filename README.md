# Deep Learning Foundations

&gt; Building tensor intuition for embodied AI. Phase 1: 2026.04 – 2026.07

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Objective

Establish intuitive understanding of tensor transformations—not mathematical derivations—to bridge deep learning algorithms into the physical world.

## Roadmap

| Phase | Timeline | Milestone |
|-------|----------|-----------|
| Foundation | 04.15–04.30 | MLP from scratch (Ch.3–4) |
| Vision Backbone | 05.01–05.31 | CNN intuition + ResNet anatomy |
| Physical Closure | 06.01–07.15 | YOLOv8 + Arduino visual servo |

## Structure
d2l-pytorch/
├── ch03_linear/           # Linear regression (completed)
├── ch04_mlp/              # MLP: forward, backward, activation
├── ch06_cnn/              # Conv layers, pooling, BatchNorm
└── ch07_modern_cnn/       # ResNet, DenseNet analysis
karpathy-nn/
└── micrograd/             # Hand-rolled autograd engine
arduino-bridge/
└── yolo_servo/            # Phase 1 capstone: visual servo loop

## Key Intuitions

| Operation | Instinct Check |
|-----------|---------------|
| `Conv2d(3,64,7,s=2)` | 3→64 channels, 7×7 kernel, halves spatial dims |
| `nn.Linear(784,256)` | 200k params, flops ≈ batch×784×256 |

## Capstone: Visual Servo

**Stack:** YOLOv8 → Python PID → PySerial → Arduino SG90
Camera ──► YOLOv8 (center error) ──► PID ──► Serial ──► Servo


**Deliverable:** `arduino_yolo_servo` repo + tracking GIF

## Progress

- [x] Ch.3 Linear regression
- [ ] Ch.4 MLP scratch implementation
- [ ] Ch.6 CNN foundations
- [ ] Ch.7 Modern CNNs
- [ ] micrograd engine
- [ ] YOLOv8 deployment
- [ ] Arduino closed-loop demo

## Reference

- D2L PyTorch Edition: Ch.3, 4, 6, 7 only
- Karpathy's micrograd (backprop from scratch)
- YOLOv8 nano for edge deployment