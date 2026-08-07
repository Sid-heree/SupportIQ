import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

def fine_tune_llm():
    print("🚀 Loading Instruction Dataset...")
    # Updated path for Colab
    dataset = load_dataset("json", data_files="/content/llm_instruction_data.json", split="train")

    def formatting_prompts_func(example):
        if isinstance(example['instruction'], list):
            output_texts = []
            for i in range(len(example['instruction'])):
                text = f"System: {example['instruction'][i]}\nUser: {example['input'][i]}\nAssistant: {example['output'][i]}"
                output_texts.append(text)
            return output_texts
        else:
            return f"System: {example['instruction']}\nUser: {example['input']}\nAssistant: {example['output']}"

    MODEL_ID = "Qwen/Qwen1.5-0.5B"
    print(f"🧠 Loading Tokenizer and Base Model ({MODEL_ID})...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # We can use standard float16 now because we have a GPU!
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        device_map="auto", 
        torch_dtype=torch.float16
    )

    print("⚙️ Setting up LoRA (Low-Rank Adaptation) Config...")
    peft_config = LoraConfig(
        r=8, 
        lora_alpha=16, 
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    print("\n🔥 Initializing SFT (Supervised Fine-Tuning) Trainer...")
    training_args = TrainingArguments(
        output_dir="/content/models/llm_supportiq",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=100, 
        save_steps=50,
        fp16=True, # GPU acceleration enabled!
        optim="adamw_torch",
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        args=training_args,
    )

    print("\n⏳ Starting LLM Fine-Tuning...")
    trainer.train()

    print("\n💾 Saving fine-tuned LoRA adapter...")
    # Save to Colab's content folder
    trainer.model.save_pretrained("/content/final_adapter")
    tokenizer.save_pretrained("/content/final_adapter")
    print("✅ LLM Fine-Tuning Complete!")

fine_tune_llm()