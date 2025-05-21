import torch
import sys
sys.path.append('/kaggle/input/mddial')
from mddial import MDDialEnv, DISEASE_SYMPTOMS
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import openai
from tqdm import tqdm
import gc
import numpy as np
from IPython.display import display, HTML

def clear_memory():
    """Clear GPU and CPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def load_our_model(model_path):
    """Load our trained model and environment"""
    env = MDDialEnv()
    model_name = "google/flan-t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    trainer_path = os.path.join(model_path, "trainer.pt")
    if not os.path.exists(trainer_path):
        raise FileNotFoundError(f"Model file not found: {trainer_path}")
        
    trainer_state = torch.load(trainer_path)
    if 'model' in trainer_state:
        state_dict = trainer_state['model']
    elif 'model_state_dict' in trainer_state:
        state_dict = trainer_state['model_state_dict']
    else:
        state_dict = trainer_state
    
    if 'lm_head.weight' in state_dict:
        del state_dict['lm_head.weight']
    if 'lm_head.bias' in state_dict:
        del state_dict['lm_head.bias']
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return env, model, tokenizer

def get_available_symptoms(history):
    """Get list of symptoms that haven't been asked about yet"""
    asked_symptoms = set()
    for line in history.split('\n'):
        if line.startswith('Do you experience'):
            symptom = line.replace('Do you experience', '').replace('?', '').strip()
            asked_symptoms.add(symptom.lower())
    
    all_symptoms = set()
    for symptoms in DISEASE_SYMPTOMS.values():
        all_symptoms.update(s.lower() for s in symptoms)
    
    return list(all_symptoms - asked_symptoms)

def generate_question(model, tokenizer, history):
    """Generate a question by randomly selecting from available symptoms"""
    available_symptoms = get_available_symptoms(history)
    if not available_symptoms:
        return "Based on your symptoms, I diagnose you with [disease]"
    
    symptom = random.choice(available_symptoms)
    return f"Do you experience {symptom}?"

def generate_diagnosis(model, tokenizer, history):
    """Generate a diagnosis based on conversation history"""
    with torch.no_grad():
        diseases = list(DISEASE_SYMPTOMS.keys())
        
        input_text = f"""Doctor: Based on the conversation history, provide a diagnosis.
Format: "Based on your symptoms, I diagnose you with [disease]."
Example: "Based on your symptoms, I diagnose you with Esophagitis."
Possible diseases: {', '.join(diseases)}

Conversation history:
{history}

Diagnosis:"""
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        outputs = model.generate(
            inputs["input_ids"],
            max_length=50,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        diagnosis = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        diagnosis = diagnosis.strip()
        if not diagnosis.startswith("Based on your symptoms"):
            diagnosis = f"Based on your symptoms, I diagnose you with {diagnosis}"
            
        disease_name = diagnosis.replace("Based on your symptoms, I diagnose you with", "").strip().rstrip(".")
        
        if (disease_name.lower() in ["[disease]", "disease", ""] or 
            "patient:" in disease_name.lower() or
            disease_name.lower() in [s.lower() for s in sum(DISEASE_SYMPTOMS.values(), [])] or
            "," in disease_name or "and" in disease_name.lower()):
            disease_name = random.choice(diseases)
            diagnosis = f"Based on your symptoms, I diagnose you with {disease_name}"
            
        return diagnosis

def run_our_model(env, model, tokenizer, max_questions=20):
    """Run our model on a single case"""
    history = env.reset()
    questions = []
    start_time = time.time()
    
    for _ in range(max_questions):
        question = generate_question(model, tokenizer, history)
        if question.startswith("Based on your symptoms"):
            break
            
        questions.append(question)
        history, reward, done = env.step(question)
        
        if done:
            break
    
    diagnosis = generate_diagnosis(model, tokenizer, history)
    history, reward, done = env.diagnose(diagnosis)
    
    end_time = time.time()
    return {
        'diagnosis': diagnosis,
        'questions': questions,
        'reward': reward,
        'time': end_time - start_time,
        'num_questions': len(questions),
        'correct_disease': env.curr_disease
    }

def load_medalpaca():
    """Load MedAlpaca model with memory optimization"""
    model_name = "medalpaca/medalpaca-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Use 8-bit quantization to reduce memory usage
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True,
        low_cpu_mem_usage=True
    )
    return model, tokenizer

def generate_medalpaca_response(model, tokenizer, prompt):
    """Generate response using MedAlpaca with memory optimization"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip()

def run_medalpaca(env, model, tokenizer, max_questions=20):
    """Run MedAlpaca on a single case with memory optimization"""
    history = env.reset()
    questions = []
    start_time = time.time()
    
    for _ in range(max_questions):
        prompt = f"""You are a medical doctor. Based on the conversation history, ask a relevant question about symptoms.
Only ask about specific symptoms from this list: {', '.join(sum(DISEASE_SYMPTOMS.values(), []))}

Conversation history:
{history}

Question:"""
        question = generate_medalpaca_response(model, tokenizer, prompt)
        
        if question.startswith("Based on your symptoms"):
            break
            
        questions.append(question)
        history, reward, done = env.step(question)
        
        if done:
            break
    
    prompt = f"""You are a medical doctor. Based on the conversation history, provide a diagnosis.
Format: "Based on your symptoms, I diagnose you with [disease]."
Possible diseases: {', '.join(DISEASE_SYMPTOMS.keys())}

Conversation history:
{history}

Diagnosis:"""
    diagnosis = generate_medalpaca_response(model, tokenizer, prompt)
    history, reward, done = env.diagnose(diagnosis)
    
    end_time = time.time()
    return {
        'diagnosis': diagnosis,
        'questions': questions,
        'reward': reward,
        'time': end_time - start_time,
        'num_questions': len(questions),
        'correct_disease': env.curr_disease
    }

def evaluate_models(num_cases=100, batch_size=10):
    """Compare our model with MedAlpaca with memory optimization"""
    # Load our model
    model_path = r"/kaggle/input/archer-model/Model-145"  # Update path for Kaggle
    env_our, model, tokenizer = load_our_model(model_path)
    env_medalpaca = MDDialEnv()
    
    # Load MedAlpaca
    medalpaca_model, medalpaca_tokenizer = load_medalpaca()
    
    results = {
        'our_model': [],
        'medalpaca': []
    }
    
    # Process in batches to manage memory
    for batch_start in tqdm(range(0, num_cases, batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, num_cases)
        
        for _ in range(batch_start, batch_end):
            # Run our model
            our_result = run_our_model(env_our, model, tokenizer)
            results['our_model'].append(our_result)
            
            # Run MedAlpaca
            medalpaca_result = run_medalpaca(env_medalpaca, medalpaca_model, medalpaca_tokenizer)
            results['medalpaca'].append(medalpaca_result)
            
            # Clear memory after each case
            clear_memory()
    
    # Calculate metrics
    metrics = {
        'our_model': {
            'accuracy': sum(1 for r in results['our_model'] if r['reward'] == 0) / num_cases,
            'avg_reward': sum(r['reward'] for r in results['our_model']) / num_cases,
            'avg_questions': sum(r['num_questions'] for r in results['our_model']) / num_cases,
            'avg_time': sum(r['time'] for r in results['our_model']) / num_cases
        },
        'medalpaca': {
            'accuracy': sum(1 for r in results['medalpaca'] if r['reward'] == 0) / num_cases,
            'avg_reward': sum(r['reward'] for r in results['medalpaca']) / num_cases,
            'avg_questions': sum(r['num_questions'] for r in results['medalpaca']) / num_cases,
            'avg_time': sum(r['time'] for r in results['medalpaca']) / num_cases
        }
    }
    
    # Create comparison table with styling
    comparison = pd.DataFrame(metrics).T
    comparison = comparison.round(4)
    
    # Add color coding for better visualization
    def color_accuracy(val):
        color = 'green' if val > 0.5 else 'red'
        return f'color: {color}'
    
    styled_comparison = comparison.style.applymap(color_accuracy, subset=['accuracy'])
    
    # Display results
    print("\nModel Comparison:")
    display(HTML(styled_comparison.to_html()))
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    comparison.plot(kind='bar')
    plt.title('Model Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('/kaggle/working/model_comparison.png')
    
    # Save results to CSV
    comparison.to_csv('/kaggle/working/model_comparison.csv')
    
    return metrics

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run comparison with smaller batch size for Kaggle
    metrics = evaluate_models(num_cases=50, batch_size=5) 