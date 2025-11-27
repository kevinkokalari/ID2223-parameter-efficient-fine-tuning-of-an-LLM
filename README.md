# ID2223 Parameter Efficient Fine-Tuning of an LLM

This project explores parameter-efficient fine-tuning (PEFT) of an open-source LLM using LoRA. Fine-tuning was performed on a Google Colab T4 GPU, with multiple checkpoints saved during training. The resulting model is deployed on Hugging Face Spaces using a modified version of Hugging Face’s Gradio chatbot template.

The fine-tuned model is used in a sentiment analysis application, classifying user input as positive, negative, or neutral, presented through a simple UI. Since Hugging Face Spaces runs CPU inference on the free tier, the trained model must be converted into a CPU-friendly format after training.

The second part of the project investigates ways to improve scalability and performance through model-centric (hyperparameters / training setup) and data-centric (dataset quality / new data) approaches, focusing on better end-to-end inference behavior.

## Tasks:

### *Task 1: Fine-tune a model for language transcription, add a UI*
1. Fine-Tune a pre-trained large language (transformer) model and build a serverless UI for using that model
2. Create a free account on huggingface.com
3. Create a free account on google.com for Colab
4. Fine-tune an existing pre-trained large language model on the FineTome Instruction Dataset
5. Build and run an inference pipeline with a Gradio UI on Hugging Face Spaces for your model.
    - Communicate the value of your model to stakeholders with an app/service that uses the fine tuned LLM to make value-added decisions
    - If you want to get the highest grade (A), come up with your own creative idea for how to allow people to use your fine tuned LLM




### *Task 2: Improve pipeline scalability and model performance*
1. Describe in the ways in which you can improve model performance are using:
   - **(a)** ***Model-centric approach*** - e.g., tune hyperparameters, change the fine-tuning model architecture, etc.
   - **(b)** ***Data-centric approach*** - identify new data sources that enable you to train a better model that one provided in the blog post
   - *If you can show results of improvement, then you get the top grade*
2. Try out fine-tuning a couple of different open-source foundation LLMs to get one that works best with your UI for inference (inference will be on CPUs, so big models will be slow).
3. You are free to use other fine-tuning frameworks, such as Axolotl of HF FineTuning - you do not have to use the provided unsloth notebook.

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
