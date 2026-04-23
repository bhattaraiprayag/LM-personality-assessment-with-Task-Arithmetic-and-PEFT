# Motivation

Large language models can absorb behavioral tendencies from the data and training strategy used to adapt them. This project studies that effect through the lens of personality-aligned modeling.

## Why this project exists

The core goal is to understand whether parameter-efficient adaptation (LoRA) and task arithmetic can steer personality-related behavior in a controllable way, while keeping model updates lightweight.

## Research goals

1. quantify shifts in personality-related outputs before and after fine-tuning,
2. compare baseline fine-tuning and LoRA-based adaptation under matched settings,
3. measure how positive and negative adapter scaling changes inventory scores,
4. build reproducible analysis pipelines that link training decisions to observed trait patterns.

## Expected outcomes

- repeatable experiment pipelines across emotion and personality splits,
- comparable scoring artifacts for BFI-10, PANAS-X, and IPIP-120,
- visual analysis of trait movement under different split definitions and LoRA scales,
- a practical base for follow-up independent research on controllable LM behavior.
