Install required libraries
huggingface_hub :

The huggingface_hub library is a Python package by Hugging Face, designed for easy access to NLP resources like pre-trained models and datasets hosted on the Hugging Face Hub. It simplifies tasks like model access, sharing, fine-tuning, and dataset downloading.
transformers :

The Transformers library, developed by Hugging Face, is a popular open-source Python library for natural language processing (NLP). It provides easy access to a wide range of pre-trained models for various NLP tasks, such as text classification, named entity recognition, text generation, and more. The library is built on top of PyTorch and TensorFlow, allowing users to easily implement state-of-the-art models like BERT, GPT, RoBERTa, and many others. With Transformers, users can fine-tune these pre-trained models on custom datasets, perform inference on new data, and even create their own models. It's widely used in both research and industry for tasks ranging from sentiment analysis to machine translation.
accelerate :

The Accelerate library is a Python package developed by Hugging Face. It's designed to optimize the training and inference performance of deep learning models, particularly those built with the Transformers library. Accelerate leverages mixed precision training, distributed training, and efficient data loading techniques to speed up the training process and improve resource utilization. By using Accelerate, users can train large-scale models faster and more efficiently, making it a valuable tool for researchers and practitioners working in natural language processing (NLP) and other deep learning domains.
bitsandbytes :

bitsandbytes enables accessible large language models via k-bit quantization for PyTorch. bitsandbytes provides three main features for dramatically reducing memory consumption for inference and training:

8-bit optimizers uses block-wise quantization to maintain 32-bit performance at a small fraction of the memory cost.
LLM.Int() or 8-bit quantization enables large language model inference with only half the required memory and without any performance degradation. This method is based on vector-wise quantization to quantize most features to 8-bits and separately treating outliers with 16-bit matrix multiplication.
QLoRA or 4-bit quantization enables large language model training with several memory-saving techniques that don’t compromise performance. This method quantizes a model to 4-bits and inserts a small set of trainable low-rank adaptation (LoRA) weights to allow training.
