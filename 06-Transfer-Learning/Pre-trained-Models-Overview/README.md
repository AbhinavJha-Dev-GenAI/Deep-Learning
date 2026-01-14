# Pre-trained Models Overview

**Transfer Learning** is the practice of taking a model trained on one task (e.g., ImageNet classification) and repurposing it for a second, related task.

## Why use Pre-trained Models?
- **Speed:** Training from scratch can take weeks/months.
- **Smaller Datasets:** Models can perform well even when you have only a few hundred images.
- **Better Performance:** Pre-trained models have already learned fundamental features (edges, shapes) that are universal.

## Popular Pre-trained Models

### Computer Vision (CV)
- **ResNet:** The industry standard workhorse.
- **EfficientNet:** Excellent accuracy-to-parameter ratio.
- **Vision Transformer (ViT):** The modern Transformer-based alternative to CNNs.
- **MobileNet:** Optimized for mobile and edge devices.

### Natural Language Processing (NLP)
- **BERT:** Bidirectional representations from Transformers.
- **GPT:** Generative Pre-trained Transformers.
- **RoBERTa:** A robustly optimized BERT approach.

## How to use them?
There are two main strategies:
1. **Feature Extraction:** Use the model as a fixed feature extractor.
2. **Fine-Tuning:** Update the weights of the pre-trained model on your new data.
