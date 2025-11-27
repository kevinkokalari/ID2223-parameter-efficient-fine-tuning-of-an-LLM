# ID2223 Parameter Efficient Fine-Tuning of an LLM

This project explores parameter-efficient fine-tuning (PEFT) of an open-source LLM using LoRA. Fine-tuning was performed on a Google Colab T4 GPU, with multiple checkpoints saved during training. The resulting model is deployed on Hugging Face Spaces using a modified version of Hugging Face’s Gradio chatbot template.

The fine-tuned model is used in a sentiment analysis application, classifying user input as positive, negative, or neutral, presented through a simple UI. Since Hugging Face Spaces runs CPU inference on the free tier, the trained model must be converted into a CPU-friendly format after training.

The second part of the project investigates ways to improve scalability and performance through model-centric (hyperparameters / training setup) and data-centric (dataset quality / new data) approaches, focusing on better end-to-end inference behavior.

## Task:
Smth

-----

# Ways to improve the fine tuning

## Model-centric adjustments
----

***Learning Rate***

***Effective Batch Size***
- Per Device Train Batch Size:
- Gradient Accumulation Steps:

***LORA Parameters***
- Rank (lora_r)
- Alpha (lora_alpha)
- Dropout (lora_dropout) 

***Max Sequence Length***

***Number of Train Epochs and Max Steps***

***Sequence and Tokenization***
- Max Sequence Length
- Packing

----
## Data-centric adjustments
----

smth
