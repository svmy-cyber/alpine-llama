# Alpine Llama: Large Language Model Fine-Tuning Toolkit

## Project Overview

Alpine Llama is a toolkit designed for fine-tuning large language models with a focus on efficient, flexible model adaptation. While the base implementation focuses on web component creation, it can be modified to work more generically with minimal adjustment to the UI's JavaScript.

## Core Fine-Tuning Capabilities

### Advanced Fine-Tuning Techniques
- Parameter-Efficient Fine-Tuning (PEFT)
- Low-Rank Adaptation (LoRA)
- Model Quantization (4-bit, 8-bit, 16-bit)
- Dynamic Layer Selection
- Adaptive Training Configurations

### Fine-Tuning Workflow
1. **Model Selection**
   - Choose from multiple base models
   - Support for various model architectures
   - Compatibility checks for hardware and model specifications

2. **Data Preparation**
   - Structured input-output example processing
   - Tokenization and dataset preparation
   - Flexible example management

3. **Training Configuration**
   - Preset training modes:
     - Fast Training (Quick iteration)
     - Balanced Training (Recommended)
     - High-Quality Training (Extensive refinement)
   - Customizable hyperparameters:
     - Learning rate
     - Batch size
     - Gradient accumulation
     - Quantization levels
     - LoRA rank and alpha

4. **Model Optimization**
   - Memory-efficient training
   - GPU-accelerated processing
   - Automated model layer selection
   - Preservation of base model knowledge

### Web Component Generation (Secondary Feature)
- HTML structure generation
- Tailwind CSS styling integration
- Alpine.js interactivity support
- Specialized training for web development tasks

## System Requirements

### Hardware
- Processor: x86_64 architecture
- RAM: 16 GB (32 GB recommended)
- Storage: 50 GB free disk space
- GPU: NVIDIA GPU with CUDA support
  - Minimum: GTX 1660 Ti
  - Recommended: RTX 3070
  - Ideal: RTX 4090 (24GB VRAM)

### Software
- Operating System: Windows 10/11 with WSL2, Ubuntu 22.04 LTS
- Python 3.11
- CUDA Toolkit 11.x or 12.x
- Git LFS
- NVIDIA GPU Drivers

## Installation

### Prerequisites
1. Enable WSL2 on Windows
2. Install Ubuntu 22.04 LTS
3. Update and upgrade system packages

### Setup
```bash
# Clone repository
git clone https://github.com/svmy-cyber/alpine-llama
cd alpine-llama

# Make scripts executable
chmod +x process.sh

# Run initial setup
./process.sh setup
```

## Usage Modes

### Fine-Tuning Execution
```bash
# Standard fine-tuning
./process.sh finetune

# Custom fine-tuning with advanced options
./process.sh finetune -c
```

### Model Interaction
```bash
# Run fine-tuned model
./process.sh run
```

## Supported Base Models
- Microsoft Phi-4
- Qwen 2.5 Coder 32B
- DeepSeek Coder V2

## Training Examples Preparation

### Example JSON Structure
```json
[
  {
    "input": "Detailed description or task",
    "output": "Corresponding model response"
  }
]
```

### Effective Example Compilation
- Recommended: 50-1000 high-quality examples
- Focus on specific domain or task
- Ensure input diversity
- Maintain consistent formatting
- Cover various complexity levels

### Example Types
- Code generation
- Specialized instructions
- Conversational patterns
- Technical documentation
- Specific domain adaptations

## Hugging Face Token Requirements

### Token Purpose
- Model access authentication
- Download permissions
- Usage tracking

### Token Acquisition
1. Create account at huggingface.co
2. Generate access token
3. Use with appropriate permissions
4. Maintain token confidentiality

## Limitations and Considerations
- Linux/GPU knowledge required
- Significant GPU memory needs
- Potentially long training times
- Model compatibility variations
- Hardware performance dependency

## Fine-Tuning Performance Factors
- Example quality/quantity
- Model base architecture
- Hardware specifications
- Training configuration
- Quantization strategy

## License

MIT License

## Disclaimer

This software is provided "as is" without warranties. Users are responsible for:
- System compatibility verification
- Licensing compliance
- Security implementations
- Thorough testing

Use at your own risk.
