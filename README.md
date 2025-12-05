# ID2223 Parameter Efficient Fine-Tuning of an LLM

This project explores parameter-efficient fine-tuning (PEFT) of an open-source LLM using LoRA. Fine-tuning was performed on a Google Colab T4 GPU, with multiple checkpoints saved during training. The resulting model is deployed on Hugging Face Spaces using a modified version of Hugging Face’s Gradio chatbot template.

The fine-tuned model is used in an email improvement application, presented in a UI. It uses sentiment analysis, a politeness classifier and the fine-tuned LLM to correct and improve emails. Since Hugging Face Spaces runs CPU inference on the free tier, the trained model must be converted into a CPU-friendly format after training.

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
# Task 1: Fine-tuned model and UI

The app.py code implements a small web app which helps us demonstrate different LLMs that we have fine-tuned. It is built using Gradio in HuggingFace spaces and behaves like an ”Email Proofreader”. The user pastes an email draft and the app evalutes it with two lightweight text classifiers: one for sentiment and one for politeness, which are shown as emoji-based indicators in the UI. These scores are then fed into the LLM prompt so the model can use them as guidance when editing. 

For generation, the app downloads a fine-tuned GGUF model from a Hugging Face model repository, and then runs inference locally through llama.cpp on the host’s CPU. The model’s system prompt is created so that the model only rewrites the email when needed while keeping its original meaning, structure and formatting. This is decided based on the original email’s grammar, clarity, sentiment, and politeness tone. The UI also exposes generation controls for the user to change at their will. Finally, the rewritten email is re-scored and shown to the user.

[HuggingFace Spaces](https://huggingface.co/spaces/greenie-sweden/chatbot-for-fine-tuned-llm)

# Task 2: Ways to improve the fine tuning

## Model-centric adjustments
----

Did SHA, following parameters...

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

The chosen model, unsloth/Llama-3.2-1B-Instruct, was initially fine-tuned with the dataset FineTome-100k. It was created by extracting a subset from the dataset acree-ai/The-Tome [1](https://huggingface.co/datasets/mlabonne/FineTome-100k). This dataset specifically targets the ability to follow instructions and is used for LLM training [2](https://huggingface.co/datasets/arcee-ai/The-Tome/blob/main/README.md).

When adopting a data centric approach and identifying a dataset for training a better model, this is of course dependent on what the end goal is. The point of fine-tuning is complementing the original model with data it was not trained on, for instance private data or data with information from after the cut-off date. Thus, the key is to figure out the purpose of the fine-tuning and what fine-tuning the original model would benefit from. The model that is fine-tuned in this assignment is the llama-3.2-1B-Instruct model. Information about the model was found here: [3](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).

The model is already tuned for instructions and supports a number of languages. However, it is trained on public data and Swedish, for instance, is not one of the officially supported languages. It is stated that the model training includes other languages, but it is likely that if the goal is to make a Swedish chatbot, it would perform better if it was fine-tuned on a Swedish corpus. For instance, the dataset swedish-sentiment-instruction-fine-tuning could be used [4](https://huggingface.co/datasets/filopedraz/swedish-sentiment-instruction-fine-tuning). 

Moreover, fine-tuning with data after the cut-off date would probably also improve performance. According to the model card, the cut-off for the model is in December 2023. Thus, fine-tuning on newer data would be beneficial. Moreover the Tome dataset, which FineTome is a subset of, seems to have been released in 2024 [5](https://www.arcee.ai/blog/arcee-ai-releases-two-open-datasets). Thus fine-tuning with a relatively new dataset would probably improve performance. One example is utilizing recent data from Common Crawl [6](https://commoncrawl.org/) or free-news-datasets by webz.io [7](https://github.com/Webhose/free-news-datasets).

Furthermore, fine-tuning using a dataset of private data would also complement the existing model. Providing it with for instance internal company data, it would be much better if the intended purpose is using it in-house or allowing customers with specific company-related questions to use it.

Depending on the initial training data for llama-3.2, it is also possible that using datasets with specific domain knowledge would improve the chatbot, enabling it to give better answers in that area. For instance, this github proposing SFT-datasets for LLMs [8](https://github.com/mlabonne/llm-datasets?tab=readme-ov-file) suggests the dataset synthetic_text_to_sql [9](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) to improve coding knowledge. In the case of the implemented UI in this assignment, utilizing sentiment analysis and an LLM to analyze and improve emails, we fine-tune the llama3.2 model on the dataset 2k_grammar_corrections [10](https://huggingface.co/datasets/ambrosfitz/2k_grammar_corrections) intended to improve grammars. This dataset has an mit license and contains sentences with grammatical errors and corresponding correct sentences.

The model fine-tuned on the grammar dataset was then evaluated using 10 sentences from its training data and 12 sentences generated using Copilot. The results show that the fine-tuned model performed better than both the original model and the model fine-tuned on the Fine-Tome dataset. The new model only got one sentence wrong, which was one of the generated sentences, resulting in an accuracy of 95%. The original model got 6 sentences wrong, 2 from the training dataset and 4 from the generated sentences, resulting in an accuracy of 73%. Thus, it can be concluded that fine-tuning on a domain-specific grammar task was successful.

----
## Fine-tuning different LLM
----

We also...

----
## References

[1] https://huggingface.co/datasets/mlabonne/FineTome-100k
[2] https://huggingface.co/datasets/arcee-ai/The-Tome/blob/main/README.md
[3] https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
[4] https://huggingface.co/datasets/filopedraz/swedish-sentiment-instruction-fine-tuning
[5] https://www.arcee.ai/blog/arcee-ai-releases-two-open-datasets
[6] https://commoncrawl.org/
[7] https://github.com/Webhose/free-news-datasets
[8] https://github.com/mlabonne/llm-datasets?tab=readme-ov-file
[9] https://huggingface.co/datasets/gretelai/synthetic_text_to_sql
[10] https://huggingface.co/datasets/ambrosfitz/2k_grammar_corrections

