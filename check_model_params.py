import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def load_model(model_path):
    """Load the model and tokenizer"""
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    return model, tokenizer

def analyze_embeddings(model):
    """Analyze embedding layers"""
    embeddings = {
        'token_embedding': model.shared.weight.shape,
        'position_embedding': model.encoder.embed_positions.weight.shape if hasattr(model.encoder, 'embed_positions') else None
    }
    return embeddings

def analyze_transformer_layers(model):
    """Analyze transformer layers"""
    layers_info = []
    
    # Analyze encoder layers
    for i, layer in enumerate(model.encoder.block):
        layer_info = {
            'layer_idx': i,
            'attention': {
                'q': layer.layer[0].SelfAttention.q.weight.shape,
                'k': layer.layer[0].SelfAttention.k.weight.shape,
                'v': layer.layer[0].SelfAttention.v.weight.shape,
                'o': layer.layer[0].SelfAttention.o.weight.shape
            },
            'mlp': {
                'fc1': layer.layer[1].DenseReluDense.wi.weight.shape,
                'fc2': layer.layer[1].DenseReluDense.wo.weight.shape
            },
            'layer_norm': {
                'ln1': layer.layer[0].layer_norm.weight.shape,
                'ln2': layer.layer[1].layer_norm.weight.shape
            }
        }
        layers_info.append(layer_info)
    
    return layers_info

def calculate_parameter_stats(model):
    """Calculate parameter statistics"""
    total_params = 0
    param_stats = defaultdict(int)
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        # Categorize parameters
        if 'shared' in name:
            param_stats['embeddings'] += param_count
        elif 'SelfAttention' in name:
            param_stats['attention'] += param_count
        elif 'DenseReluDense' in name:
            param_stats['mlp'] += param_count
        elif 'layer_norm' in name:
            param_stats['layer_norm'] += param_count
        else:
            param_stats['other'] += param_count
    
    return total_params, param_stats

def visualize_parameter_distribution(param_stats):
    """Create visualization of parameter distribution"""
    plt.figure(figsize=(10, 6))
    components = list(param_stats.keys())
    values = list(param_stats.values())
    
    plt.bar(components, values)
    plt.title('Parameter Distribution Across Components')
    plt.xlabel('Component')
    plt.ylabel('Number of Parameters')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('parameter_distribution.png')
    plt.close()

def analyze_weight_distributions(model):
    """Analyze weight distributions in different components"""
    distributions = defaultdict(list)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'SelfAttention' in name:
                distributions['attention'].extend(param.detach().cpu().numpy().flatten())
            elif 'DenseReluDense' in name:
                distributions['mlp'].extend(param.detach().cpu().numpy().flatten())
            elif 'layer_norm' in name:
                distributions['layer_norm'].extend(param.detach().cpu().numpy().flatten())
    
    # Plot distributions
    plt.figure(figsize=(15, 5))
    for i, (component, values) in enumerate(distributions.items(), 1):
        plt.subplot(1, 3, i)
        sns.histplot(values, bins=50)
        plt.title(f'{component} Weight Distribution')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('weight_distributions.png')
    plt.close()

def main():
    # Load model
    model_path = "path_to_your_model"  # Update this path
    model, tokenizer = load_model(model_path)
    
    # Analyze embeddings
    embeddings = analyze_embeddings(model)
    print("\nEmbedding Analysis:")
    for name, shape in embeddings.items():
        if shape:
            print(f"{name}: {shape}")
    
    # Analyze transformer layers
    layers_info = analyze_transformer_layers(model)
    print("\nTransformer Layer Analysis:")
    for layer in layers_info[:2]:  # Print first 2 layers as example
        print(f"\nLayer {layer['layer_idx']}:")
        print("Attention shapes:", layer['attention'])
        print("MLP shapes:", layer['mlp'])
        print("Layer Norm shapes:", layer['layer_norm'])
    
    # Calculate parameter statistics
    total_params, param_stats = calculate_parameter_stats(model)
    print(f"\nTotal Parameters: {total_params:,}")
    print("\nParameter Distribution:")
    for component, count in param_stats.items():
        print(f"{component}: {count:,} ({count/total_params*100:.2f}%)")
    
    # Create visualizations
    visualize_parameter_distribution(param_stats)
    analyze_weight_distributions(model)
    
    # Save detailed report
    with open('model_analysis_report.txt', 'w') as f:
        f.write("Model Analysis Report\n")
        f.write("===================\n\n")
        f.write(f"Total Parameters: {total_params:,}\n\n")
        f.write("Parameter Distribution:\n")
        for component, count in param_stats.items():
            f.write(f"{component}: {count:,} ({count/total_params*100:.2f}%)\n")
        f.write("\nEmbedding Analysis:\n")
        for name, shape in embeddings.items():
            if shape:
                f.write(f"{name}: {shape}\n")
        f.write("\nTransformer Layer Analysis (First Layer):\n")
        f.write(str(layers_info[0]))

if __name__ == "__main__":
    main() 