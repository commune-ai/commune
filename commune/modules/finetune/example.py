from finetuning_suite import FineTuner

model_id="google/flan-t5-xxl"

dataset_name="samsum"


finetune = FineTuner(
    model_id=model_id,
    dataset_name=dataset_name,
    max_length=150,
    lora_r=16,
    lora_alpha=32,
    quantize=True
)


prompt_text = "Tell me about yourself"
generated_text = finetune(prompt_text)
print(generated_text)