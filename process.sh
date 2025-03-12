#!/bin/bash

# ========================================================
# LLM Model Fine-tuning and Execution Tool
# For WSL Ubuntu 22.04 LTS
# ========================================================

# Global variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/process.log"
ERROR_LOG="${SCRIPT_DIR}/errors.log"
CREDENTIALS_FILE="${SCRIPT_DIR}/.credentials"
VENV_DIR="${SCRIPT_DIR}/venv"
MODELS_DIR="${SCRIPT_DIR}/models"
DATA_DIR="${SCRIPT_DIR}/data"
CACHE_DIR="${HOME}/.cache/llm-finetuning"
EXAMPLES_FILE="${DATA_DIR}/examples.json"
SAMPLE_EXAMPLES='[{"input": "Create a single html tag", "output":"<html>example output</html>"}]'
OLLAMA_MODEL=""
HF_MODEL=""

# Cache directories structure
HF_CACHE_DIR="${CACHE_DIR}/huggingface"
OLLAMA_CACHE_DIR="${CACHE_DIR}/ollama"
PIP_CACHE_DIR="${CACHE_DIR}/pip"
VENDOR_CACHE_DIR="${CACHE_DIR}/vendor"

# Python script for fine-tuning
FINETUNE_SCRIPT="${SCRIPT_DIR}/finetune.py"

# ========================================================
# Utility Functions
# ========================================================

# Function to log messages
log() {
    local message="$1"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo -e "[${timestamp}] ${message}" | tee -a "${LOG_FILE}"
}

# Enhanced error logging with automatic context detection
log_error() {
    local message="$1"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    
    # Get calling function name and line number
    local function_name="${FUNCNAME[1]:-main}"
    local line_number="${BASH_LINENO[0]:-unknown}"
    local file_name="$(basename "${BASH_SOURCE[1]:-$0}")"
    
    # Format error message with context
    local error_context="${file_name}:${function_name}():${line_number}"
    echo -e "[${timestamp}] ERROR (${error_context}): ${message}" | tee -a "${ERROR_LOG}"
    
    # Add suggestion for common errors if recognized patterns are found
    if [[ "$message" == *"command not found"* ]]; then
        echo -e "HINT: This may be a missing dependency. Try running './process.sh setup' first." | tee -a "${ERROR_LOG}"
    elif [[ "$message" == *"permission denied"* ]]; then
        echo -e "HINT: This may be a permissions issue. Check file permissions or try using sudo." | tee -a "${ERROR_LOG}"
    fi
}

# Function to print progress messages
print_progress() {
    local message="$1"
    echo -e "\e[1;34m[*] ${message}\e[0m"
    log "${message}"
}

# Function to print success messages
print_success() {
    local message="$1"
    echo -e "\e[1;32m[✓] ${message}\e[0m"
    log "${message}"
}

# Function to print error messages
print_error() {
    local message="$1"
    echo -e "\e[1;31m[✗] ${message}\e[0m"
    log_error "${message}"
}

# Function to execute a command with error handling
execute_cmd() {
    local cmd="$1"
    local error_msg="$2"
    
    log "Executing: ${cmd}"
    
    # Execute the command, capturing output and redirecting to log file
    local output
    if ! output=$(eval "${cmd}" 2>&1); then
        print_error "${error_msg}"
        log_error "${error_msg}\nCommand: ${cmd}\nOutput: ${output}"
        return 1
    fi
    
    # Log the output for debugging purposes
    log "Command output: ${output}"
    return 0
}

# Ask for sudo password
get_sudo_password() {
    if [[ -z "${SUDO_PASSWORD}" ]]; then
        if [[ -f "${CREDENTIALS_FILE}" ]] && grep -q "SUDO_PASSWORD=" "${CREDENTIALS_FILE}"; then
            source "${CREDENTIALS_FILE}"
        else
            read -s -p "Enter your sudo password: " SUDO_PASSWORD
            echo
            echo "SUDO_PASSWORD=${SUDO_PASSWORD}" > "${CREDENTIALS_FILE}"
            chmod 600 "${CREDENTIALS_FILE}"
        fi
    fi
}

# Function to run a command with sudo
sudo_cmd() {
    local cmd="$1"
    echo "${SUDO_PASSWORD}" | sudo -S bash -c "${cmd}" >> "${LOG_FILE}" 2>> "${ERROR_LOG}"
    return $?
}

# Check if a package is installed
is_package_installed() {
    local package="$1"
    dpkg -l "${package}" &> /dev/null
    return $?
}

# Check if a Python package is installed in the virtual environment
is_pip_package_installed() {
    local package="$1"
    if [[ -d "${VENV_DIR}" ]]; then
        "${VENV_DIR}/bin/pip" list | grep -q "${package}"
        return $?
    else
        return 1
    fi
}

select_models() {
    # Define complete model data in a structured way
    local -A MODEL_DATA=(
        [1]="phi4 microsoft/phi-4 Microsoft's Phi-4 language model"
        [2]="qwen2.5-coder:32b Qwen/Qwen2.5-Coder-32B Alibaba's 32B code-specialized model"
        [3]="deepseek-coder-v2 deepseek-ai/DeepSeek-Coder-V2-Base DeepSeek's code generation model"
    )
    
    # Print model options with descriptions
    print_progress "Select model:"
    echo "Available models:"
    echo "----------------"
    
    for i in "${!MODEL_DATA[@]}"; do
        IFS=' ' read -r ollama_name hf_name description <<< "${MODEL_DATA[$i]}"
        printf "%d. %-20s - %s\n" "$i" "$ollama_name" "$description"
    done
    echo "----------------"
    
    # Get user selection
    read -p "Enter your choice (1-${#MODEL_DATA[@]}, default: 1): " choice
    choice=${choice:-1}
    
    # Validate input
    if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${#MODEL_DATA[@]}" ]; then
        print_error "Invalid selection. Using default: phi4"
        choice=1
    fi
    
    # Extract model information
    IFS=' ' read -r OLLAMA_MODEL HF_MODEL description <<< "${MODEL_DATA[$choice]}"
    
    print_progress "Selected: OLLAMA=${OLLAMA_MODEL}, HF=${HF_MODEL}"
}

# Function to create the required directory structure
create_directories() {
    print_progress "Creating directory structure"
    
    # Define directory lists with their permissions
    local dirs_755=(
        "${VENV_DIR}"
        "${MODELS_DIR}"
        "${DATA_DIR}"
        "${CACHE_DIR}"
        "${HF_CACHE_DIR}"
        "${OLLAMA_CACHE_DIR}"
        "${PIP_CACHE_DIR}"
        "${VENDOR_CACHE_DIR}"
    )
    
    local files_644=(
        "${LOG_FILE}"
        "${ERROR_LOG}"
    )
    
    # Create directories
    for dir in "${dirs_755[@]}"; do
        mkdir -p "$dir"
        chmod 755 "$dir"
    done
    
    # Create or touch log files
    for file in "${files_644[@]}"; do
        touch "$file"
        chmod 644 "$file"
    done
    
    print_success "Directory structure created successfully"
}

# Function to create sample examples.json if it doesn't exist
create_sample_examples() {
    if [[ ! -f "${EXAMPLES_FILE}" ]]; then
        print_progress "Creating sample examples.json file"
        echo "${SAMPLE_EXAMPLES}" > "${EXAMPLES_FILE}"
        print_success "Sample examples.json file created"
    else
        log "examples.json file already exists, skipping creation"
    fi
}

    # Check for CUDA capability
check_cuda() {
    print_progress "Checking CUDA capability"
    
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "NVIDIA driver not found. Please install NVIDIA drivers for your GPU."
        return 1
    fi
    
    if ! nvidia-smi &> /dev/null; then
        print_error "Unable to communicate with NVIDIA driver. Please check your driver installation."
        return 1
    fi
    
    local cuda_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
    print_success "CUDA is working. NVIDIA driver version: ${cuda_version}"
    return 0
}

# Create Python virtual environment
create_venv() {
    if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
        print_progress "Creating Python virtual environment"
        
        if ! python3.11 -m venv "${VENV_DIR}" 2>> "${ERROR_LOG}"; then
            print_error "Failed to create virtual environment with python3.11"
            
            # Fallback to python3
            print_progress "Trying to create virtual environment with python3"
            if ! python3 -m venv "${VENV_DIR}" 2>> "${ERROR_LOG}"; then
                print_error "Failed to create virtual environment with python3"
                return 1
            fi
        fi
        
        print_success "Python virtual environment created successfully"
    else
        print_success "Python virtual environment already exists"
    fi
    
    # Ensure pip is up to date in the virtual environment
    print_progress "Updating pip in virtual environment"
    if ! "${VENV_DIR}/bin/pip" install --upgrade pip -q 2>> "${ERROR_LOG}"; then
        print_error "Failed to update pip in virtual environment"
        return 1
    fi
    
    print_success "Virtual environment is ready"
    return 0
}

# ========================================================
# Setup Function
# ========================================================

setup() {
    log "Starting setup process"
    print_progress "Starting setup process"
    
    # Get sudo password
    get_sudo_password
    
    # Create required directories and files
    create_directories
    create_sample_examples
    
    # Update package lists
    print_progress "Updating package lists"
    if ! sudo_cmd "apt-get update -qq"; then
        print_error "Failed to update package lists"
        return 1
    fi
    
    # Install Python 3.11 and related packages
    print_progress "Installing Python 3.11 and related packages"
    if ! is_package_installed "python3.11"; then
        sudo_cmd "apt-get install -y software-properties-common"
        sudo_cmd "add-apt-repository -y ppa:deadsnakes/ppa"
        sudo_cmd "apt-get update -qq"
        sudo_cmd "apt-get install -y python3.11 python3.11-venv python3.11-distutils python3-venv"
    else
        print_success "Python 3.11 is already installed"
    fi
    
    # Create and activate Python virtual environment
    if ! create_venv; then
        print_error "Failed to set up Python virtual environment"
        return 1
    fi
    
    # Install PyTorch with CUDA support
    if ! is_pip_package_installed "torch"; then
        print_progress "Installing PyTorch with CUDA support"
        "${VENV_DIR}/bin/pip" install --cache-dir="${PIP_CACHE_DIR}" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 2>> "${ERROR_LOG}"
        if [[ $? -ne 0 ]]; then
            print_error "Failed to install PyTorch with CUDA support"
            return 1
        fi
        print_success "PyTorch installed successfully"
    else
        print_success "PyTorch is already installed"
    fi
    
    # Install CUDA WSL tools if not already working
    if ! check_cuda; then
        print_progress "Installing CUDA WSL tools"
        local cuda_keyring_file="${VENDOR_CACHE_DIR}/cuda-keyring_1.1-1_all.deb"
        
        if [[ ! -f "${cuda_keyring_file}" ]]; then
            wget -q -O "${cuda_keyring_file}" "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
        fi
        
        sudo_cmd "dpkg -i ${cuda_keyring_file}"
        sudo_cmd "apt-get update -qq"
        sudo_cmd "apt-get install -y cuda-toolkit-12-1"
        
        if ! check_cuda; then
            print_error "Failed to install CUDA. Please check your GPU and driver compatibility."
            return 1
        fi
    fi
    
    # Install cuDNN libraries
    print_progress "Installing cuDNN libraries"
    if ! is_package_installed "libcudnn8"; then
        sudo_cmd "apt-get install -y libcudnn8 libcudnn8-dev"
    else
        print_success "cuDNN libraries are already installed"
    fi
    
    # Install Git LFS
    print_progress "Installing Git LFS"
    if ! command -v git-lfs &> /dev/null; then
        sudo_cmd "apt-get install -y git-lfs"
        execute_cmd "git lfs install" "Failed to install Git LFS"
    else
        print_success "Git LFS is already installed"
    fi
    
    # Install HuggingFace Transformers and PEFT
    print_progress "Installing Transformers and PEFT"
    if ! is_pip_package_installed "transformers"; then
        "${VENV_DIR}/bin/pip" install --cache-dir="${PIP_CACHE_DIR}" transformers 2>> "${ERROR_LOG}"
        if [[ $? -ne 0 ]]; then
            print_error "Failed to install Transformers"
            return 1
        fi
    else
        print_success "Transformers is already installed"
    fi
    
    if ! is_pip_package_installed "peft"; then
        "${VENV_DIR}/bin/pip" install --cache-dir="${PIP_CACHE_DIR}" peft 2>> "${ERROR_LOG}"
        if [[ $? -ne 0 ]]; then
            print_error "Failed to install PEFT"
            return 1
        fi
    else
        print_success "PEFT is already installed"
    fi
    
    # Install protobuf first (required for SentencePiece) - ensuring specific version
    print_progress "Installing protobuf"
    # Force reinstall protobuf to ensure correct version
    "${VENV_DIR}/bin/pip" uninstall -y protobuf >> "${LOG_FILE}" 2>> "${ERROR_LOG}" || true
    "${VENV_DIR}/bin/pip" install --cache-dir="${PIP_CACHE_DIR}" protobuf==3.20.3 --force-reinstall >> "${LOG_FILE}" 2>> "${ERROR_LOG}"
    if [[ $? -ne 0 ]]; then
        print_error "Failed to install protobuf"
        return 1
    fi
    print_success "protobuf is installed"
    
    # Verify protobuf installation
    print_progress "Verifying protobuf installation"
    if ! "${VENV_DIR}/bin/python" -c "import google.protobuf; print('Protobuf is properly installed')" >> "${LOG_FILE}" 2>> "${ERROR_LOG}"; then
        print_error "Protobuf verification failed. This may cause issues with SentencePiece."
        
        # Try alternative installation method
        print_progress "Trying alternative protobuf installation method"
        sudo_cmd "apt-get install -y python3-protobuf"
        "${VENV_DIR}/bin/pip" install --cache-dir="${PIP_CACHE_DIR}" --upgrade "setuptools<60.0.0" >> "${LOG_FILE}" 2>> "${ERROR_LOG}"
        "${VENV_DIR}/bin/pip" install --cache-dir="${PIP_CACHE_DIR}" --upgrade protobuf==3.20.3 >> "${LOG_FILE}" 2>> "${ERROR_LOG}"
        
        # Verify again
        if ! "${VENV_DIR}/bin/python" -c "import google.protobuf; print('Protobuf is properly installed')" >> "${LOG_FILE}" 2>> "${ERROR_LOG}"; then
            print_error "Alternative protobuf installation failed. Continuing but fine-tuning may fail."
        else
            print_success "Alternative protobuf installation successful"
        fi
    else
        print_success "Protobuf verified successfully"
    fi
    
    # Install SentencePiece with proper dependencies
    print_progress "Installing SentencePiece"
    # Force reinstall to ensure correct version with dependencies
    "${VENV_DIR}/bin/pip" uninstall -y sentencepiece >> "${LOG_FILE}" 2>> "${ERROR_LOG}" || true
    
    # Try to install from PyPI first with explicit dependencies
    if ! "${VENV_DIR}/bin/pip" install --cache-dir="${PIP_CACHE_DIR}" "sentencepiece>=0.1.99" >> "${LOG_FILE}" 2>> "${ERROR_LOG}"; then
        print_progress "PyPI installation failed, trying with specific wheel file"
        
        # Fallback to specific wheel if pip install fails
        local sentencepiece_file="${VENDOR_CACHE_DIR}/sentencepiece-0.1.99-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
        
        if [[ ! -f "${sentencepiece_file}" ]]; then
            print_progress "Downloading SentencePiece wheel"
            local sentencepiece_url="https://github.com/google/sentencepiece/releases/download/v0.1.99/sentencepiece-0.1.99-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
            wget -q -O "${sentencepiece_file}" "${sentencepiece_url}" >> "${LOG_FILE}" 2>> "${ERROR_LOG}"
        fi
        
        "${VENV_DIR}/bin/pip" install "${sentencepiece_file}" >> "${LOG_FILE}" 2>> "${ERROR_LOG}"
        if [[ $? -ne 0 ]]; then
            print_error "Failed to install SentencePiece from wheel. Trying build from source."
            
            # Try installing from source as a last resort
            sudo_cmd "apt-get install -y cmake build-essential pkg-config libgoogle-perftools-dev"
            "${VENV_DIR}/bin/pip" install --cache-dir="${PIP_CACHE_DIR}" sentencepiece --no-binary :all: >> "${LOG_FILE}" 2>> "${ERROR_LOG}"
            
            if [[ $? -ne 0 ]]; then
                print_error "Failed to install SentencePiece from source"
                return 1
            else
                print_success "SentencePiece installed from source"
            fi
        else
            print_success "SentencePiece installed from wheel"
        fi
    else
        print_success "SentencePiece installed from PyPI"
    fi
    
    # Verify SentencePiece installation
    print_progress "Verifying SentencePiece installation"
    if ! "${VENV_DIR}/bin/python" -c "import sentencepiece; print('SentencePiece is properly installed')" >> "${LOG_FILE}" 2>> "${ERROR_LOG}"; then
        print_error "SentencePiece verification failed. This will cause issues with tokenization."
        return 1
    else
        print_success "SentencePiece verified successfully"
    fi
    
    # Install additional required packages
    print_progress "Installing additional required packages"
    "${VENV_DIR}/bin/pip" install --cache-dir="${PIP_CACHE_DIR}" accelerate bitsandbytes datasets safetensors 2>> "${ERROR_LOG}"
    
    # Install Ollama
    print_progress "Installing Ollama"
    if ! command -v ollama &> /dev/null; then
        local ollama_file="${VENDOR_CACHE_DIR}/ollama-linux-amd64"
        
        if [[ ! -f "${ollama_file}" ]]; then
            wget -q -O "${ollama_file}" "https://ollama.com/download/ollama-linux-amd64"
        fi
        
        sudo_cmd "install -m 755 ${ollama_file} /usr/local/bin/ollama"
        
        if ! command -v ollama &> /dev/null; then
            print_error "Failed to install Ollama"
            return 1
        fi
    else
        print_success "Ollama is already installed"
    fi
    
    # Create the fine-tuning Python script
    create_finetune_script
    
    print_success "Setup completed successfully"
    return 0
}

# Create fine-tuning script
create_finetune_script() {
    print_progress "Creating fine-tuning script"
    
    cat > "${FINETUNE_SCRIPT}" << 'EOL'
#!/usr/bin/env python3
import os
import argparse
import json
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# Add error handling decorator
def error_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"ERROR: {str(e)}")
            import traceback
            print(traceback.format_exc())
            sys.exit(1)
    return wrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLM models with PEFT")
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--ollama_model", type=str, required=True, help="Ollama model name")
    parser.add_argument("--examples_file", type=str, required=True, help="Path to examples.json file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the model")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for models")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for model generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for model generation")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k for model generation")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.", 
                        help="System prompt for the model")
    return parser.parse_args()

def prepare_dataset(examples_file):
    with open(examples_file, 'r') as f:
        examples = json.load(f)
    
    # Prepare data for training
    dataset_dict = {
        "input": [example["input"] for example in examples],
        "output": [example["output"] for example in examples]
    }
    
    return Dataset.from_dict(dataset_dict)

def tokenize_function(examples, tokenizer, max_length=512):
    inputs = tokenizer(examples["input"], padding="max_length", truncation=True, max_length=max_length)
    outputs = tokenizer(examples["output"], padding="max_length", truncation=True, max_length=max_length)
    
    # Prepare the labels
    labels = outputs["input_ids"].copy()
    # Set padding tokens to -100 so they are not included in loss computation
    labels = [[-100 if token == tokenizer.pad_token_id else token for token in seq] for seq in labels]
    
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels
    }

def create_ollama_model_file(args, output_dir):
    model_file_content = f"""
FROM {args.ollama_model}

# Set generation parameters
PARAMETER temperature {args.temperature}
PARAMETER top_p {args.top_p}
PARAMETER top_k {args.top_k}

# Set system prompt
SYSTEM {args.system_prompt}
"""
    
    model_file_path = os.path.join(output_dir, "Modelfile")
    with open(model_file_path, 'w') as f:
        f.write(model_file_content.strip())
    
    return model_file_path

@error_handler
def main():
    args = parse_args()
    
    print(f"Loading model: {args.model_name}")
    
    # Set up comprehensive error handling for dependencies
    # Check for protobuf installation
    try:
        import google.protobuf
        print(f"Protobuf is properly installed (version: {google.protobuf.__version__})")
    except ImportError as e:
        print(f"WARNING: Google protobuf error: {str(e)}. Installing it now...")
        import subprocess
        try:
            # Try a specific version known to work with SentencePiece
            subprocess.run(["pip", "install", "--force-reinstall", "protobuf==3.20.3"], check=True)
            print("Installed protobuf 3.20.3. Trying to import again...")
            import google.protobuf
            print(f"Protobuf is now installed (version: {google.protobuf.__version__})")
        except Exception as install_error:
            print(f"ERROR installing protobuf: {str(install_error)}")
            print("Trying to continue anyway, but this may cause errors...")
    
    # Check for SentencePiece installation
    try:
        import sentencepiece
        print(f"SentencePiece is properly installed (version: {sentencepiece.__version__})")
    except ImportError as e:
        print(f"WARNING: SentencePiece error: {str(e)}. Installing it now...")
        import subprocess
        try:
            subprocess.run(["pip", "install", "--force-reinstall", "sentencepiece>=0.1.99"], check=True)
            print("Installed SentencePiece. Trying to import again...")
            import sentencepiece
            print(f"SentencePiece is now installed (version: {sentencepiece.__version__})")
        except Exception as install_error:
            print(f"ERROR installing SentencePiece: {str(install_error)}")
            raise ImportError("Critical dependency SentencePiece could not be installed. Cannot continue.") 
    
    # Create BitsAndBytesConfig for quantization (replacing deprecated load_in_8bit)
    from transformers import BitsAndBytesConfig
    
    # Check GPU memory - use 4-bit quantization if available
    try:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # in GB
        print(f"GPU memory detected: {gpu_memory:.2f} GB")
        
        # Use 4-bit quantization for larger models or limited GPU memory
        if gpu_memory < 24 or "70b" in args.model_name.lower() or "65b" in args.model_name.lower():
            print("Using 4-bit quantization for better memory efficiency")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            # Use 8-bit quantization for better quality with sufficient memory
            print("Using 8-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
    except Exception as e:
        print(f"Error detecting GPU memory: {str(e)}. Defaulting to 8-bit quantization.")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
    
    # Load the model and tokenizer with updated configuration
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=args.cache_dir,
        trust_remote_code=True
    )
    
    # First try with fast tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Fast tokenizer failed with error: {str(e)}")
        print("Falling back to slow tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA - dynamically determine target modules based on model architecture
    target_modules = []
    
    # Check model architecture to set appropriate target modules
    for name, _ in model.named_modules():
        if any(module_type in name for module_type in ["q_proj", "v_proj", "k_proj", "o_proj"]):
            if "q_proj" not in target_modules and "q_proj" in name:
                target_modules.append("q_proj")
            if "v_proj" not in target_modules and "v_proj" in name:
                target_modules.append("v_proj")
            if "k_proj" not in target_modules and "k_proj" in name:
                target_modules.append("k_proj")
            if "o_proj" not in target_modules and "o_proj" in name:
                target_modules.append("o_proj")
        
        if any(module_type in name for module_type in ["gate_proj", "up_proj", "down_proj"]):
            if "gate_proj" not in target_modules and "gate_proj" in name:
                target_modules.append("gate_proj")
            if "up_proj" not in target_modules and "up_proj" in name:
                target_modules.append("up_proj")
            if "down_proj" not in target_modules and "down_proj" in name:
                target_modules.append("down_proj")
    
    # If no modules were found, use a default set for common architectures
    if not target_modules:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    print(f"Using LoRA target modules: {target_modules}")
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules
    )
    
    model = get_peft_model(model, lora_config)
    
    # Load and prepare dataset
    dataset = prepare_dataset(args.examples_file)
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_steps=10,
        optim="paged_adamw_8bit",
        report_to=None,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        load_best_model_at_end=False,
        save_total_limit=3,  # Keep only the 3 best checkpoints
        resume_from_checkpoint=True 
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=None 
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the model
    print(f"Saving fine-tuned model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Create Ollama model file
    model_file_path = create_ollama_model_file(args, args.output_dir)
    print(f"Created Ollama model file at {model_file_path}")
    
    print("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()
EOL
    
    chmod +x "${FINETUNE_SCRIPT}"
    print_success "Fine-tuning script created successfully"
}

# ========================================================
# Fine-tuning Function
# ========================================================

finetune() {
    log "Starting fine-tuning process"
    print_progress "Starting fine-tuning process"
    
    # Get sudo password
    get_sudo_password
    
    # Check if we're using custom mode
    local custom_mode=false
    if [[ "$1" == "-c" ]]; then
        custom_mode=true
        print_progress "Running in custom fine-tuning mode"
    fi
    
    # Select models
    select_models

    # Create a detailed environment check
    perform_environment_check
    
    # Validate inputs
    if [[ -z "${OLLAMA_MODEL}" || -z "${HF_MODEL}" ]]; then
        print_error "Model names cannot be empty"
        return 1
    fi
    
    # Create model directory
    local model_dir="${MODELS_DIR}/${OLLAMA_MODEL}"
    mkdir -p "${model_dir}"
    
    # Pull the base model from Ollama
    print_progress "Pulling base model from Ollama: ${OLLAMA_MODEL}"
    if ! ollama pull "${OLLAMA_MODEL}" >> "${LOG_FILE}" 2>> "${ERROR_LOG}"; then
        print_error "Failed to pull model from Ollama"
        return 1
    fi
    
    # Configure HuggingFace token
    configure_huggingface_token
    
    # Initialize default settings
    local temperature=0.7
    local top_p=0.9
    local top_k=50
    local system_prompt="You are Mistral AI's assistant. You are honest, helpful and harmless."
    
    # Custom fine-tuning parameters (only used in custom mode)
    local lora_rank=16
    local lora_alpha=32
    local lora_dropout=0.05
    local learning_rate="2e-4"
    local batch_size=4
    local grad_acc_steps=4
    local num_epochs=3
    local warmup_steps=50
    local quant_bits=8
    local preset_name="Balanced"
    
    if [[ "${custom_mode}" == true ]]; then
        # Run the custom fine-tuning configuration flow
        configure_custom_finetune_settings
    else
        # Get model configuration with validation and Mistral-specific defaults
        configure_basic_finetune_settings
    fi
    
    # Validate examples.json exists
    if [[ ! -f "${EXAMPLES_FILE}" ]]; then
        print_error "examples.json file not found at ${EXAMPLES_FILE}"
        return 1
    fi
    
    # Determine which finetune script to use
    local finetune_script_path
    if [[ "${custom_mode}" == true ]]; then
        finetune_script_path="${SCRIPT_DIR}/custom_finetune.py"
        create_custom_finetune_script "${finetune_script_path}"
    else
        finetune_script_path="${FINETUNE_SCRIPT}"
    fi
    
    # Create wrapper script
    local wrapper_script="${SCRIPT_DIR}/finetune_wrapper.py"
    create_finetune_wrapper_script "${wrapper_script}"
    
    # Fine-tune the model using the wrapper script
    print_progress "Starting fine-tuning process"
    
    if [[ "${custom_mode}" == true ]]; then
        "${VENV_DIR}/bin/python" "${wrapper_script}" "${finetune_script_path}" \
            --model_name "${HF_MODEL}" \
            --ollama_model "${OLLAMA_MODEL}" \
            --examples_file "${EXAMPLES_FILE}" \
            --output_dir "${model_dir}/trained-model" \
            --cache_dir "${HF_CACHE_DIR}" \
            --temperature "${temperature}" \
            --top_p "${top_p}" \
            --top_k "${top_k}" \
            --system_prompt "${system_prompt}" \
            --lora_rank "${lora_rank}" \
            --lora_alpha "${lora_alpha}" \
            --lora_dropout "${lora_dropout}" \
            --learning_rate "${learning_rate}" \
            --batch_size "${batch_size}" \
            --grad_acc_steps "${grad_acc_steps}" \
            --num_epochs "${num_epochs}" \
            --warmup_steps "${warmup_steps}" \
            --quant_bits "${quant_bits}" \
            2>&1 | tee -a "${LOG_FILE}"
    else
        "${VENV_DIR}/bin/python" "${wrapper_script}" "${finetune_script_path}" \
            --model_name "${HF_MODEL}" \
            --ollama_model "${OLLAMA_MODEL}" \
            --examples_file "${EXAMPLES_FILE}" \
            --output_dir "${model_dir}/trained-model" \
            --cache_dir "${HF_CACHE_DIR}" \
            --temperature "${temperature}" \
            --top_p "${top_p}" \
            --top_k "${top_k}" \
            --system_prompt "${system_prompt}" \
            2>&1 | tee -a "${LOG_FILE}"
    fi
    
    # Check the exit status - need to use PIPESTATUS since we're using tee
    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        print_error "Fine-tuning process failed"
        return 1
    fi
    
    # Create Ollama model from the fine-tuned model
    print_progress "Creating Ollama model from fine-tuned model"
    
    cd "${model_dir}/trained-model" || {
        print_error "Failed to change directory to ${model_dir}/trained-model"
        return 1
    }
    
    if ! ollama create "${OLLAMA_MODEL}-ft" -f "Modelfile" >> "${LOG_FILE}" 2>> "${ERROR_LOG}"; then
        print_error "Failed to create Ollama model"
        return 1
    fi
    
    if [[ "${custom_mode}" == true ]]; then
        print_success "Custom fine-tuning completed successfully with ${preset_name} preset"
    else
        print_success "Fine-tuning completed successfully"
    fi
    
    print_success "Your fine-tuned model is available as: ${OLLAMA_MODEL}-ft"
    return 0
}

perform_environment_check() {
    print_progress "Performing environment check before fine-tuning"
    if "${VENV_DIR}/bin/python" -c "
import sys
import pkg_resources
import platform
import os

print('Python version:', sys.version)
print('Platform:', platform.platform())

critical_packages = ['torch', 'transformers', 'peft', 'protobuf', 'sentencepiece']
print('\\nCritical package versions:')
for package in critical_packages:
    try:
        version = pkg_resources.get_distribution(package).version
        print(f'- {package}: {version}')
    except pkg_resources.DistributionNotFound:
        print(f'- {package}: NOT INSTALLED')

# Test cuda availability
import torch
print('\\nCUDA information:')
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU count:', torch.cuda.device_count())
    print('GPU name:', torch.cuda.get_device_name(0))
    print('GPU memory:', torch.cuda.get_device_properties(0).total_memory / (1024**3), 'GB')

# Test Protobuf
print('\\nProtobuf test:')
try:
    import google.protobuf
    print('Protobuf import successful, version:', google.protobuf.__version__)
except ImportError as e:
    print('Protobuf import failed:', e)

# Test SentencePiece
print('\\nSentencePiece test:')
try:
    import sentencepiece
    print('SentencePiece import successful, version:', sentencepiece.__version__)
except ImportError as e:
    print('SentencePiece import failed:', e)

print('\\nEnvironment variables:')
for var in ['CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH', 'PATH', 'PYTHONPATH']:
    print(f'{var}:', os.environ.get(var, 'Not set'))
" >> "${LOG_FILE}" 2>> "${ERROR_LOG}"; then
        print_success "Environment check completed"
    else
        print_error "Environment check failed. Check logs for details."
        # Continue anyway as this is just informational
    fi
}

configure_huggingface_token() {
    # Get HuggingFace token
    local hf_token=""
    if [[ -f "${CREDENTIALS_FILE}" ]] && grep -q "HF_TOKEN=" "${CREDENTIALS_FILE}"; then
        source "${CREDENTIALS_FILE}"
        hf_token="${HF_TOKEN}"
    else
        read -s -p "Enter your HuggingFace token: " hf_token
        echo
        echo "HF_TOKEN=${hf_token}" >> "${CREDENTIALS_FILE}"
        chmod 600 "${CREDENTIALS_FILE}"
    fi
    
    # Configure HuggingFace token
    export HUGGING_FACE_HUB_TOKEN="${hf_token}"
    print_progress "Configuring HuggingFace token"
    echo "n" | "${VENV_DIR}/bin/huggingface-cli" login --token "${hf_token}" >> "${LOG_FILE}" 2>> "${ERROR_LOG}"
}

# Function to get validated user input
get_user_input() {
    local prompt="$1"           # Display prompt
    local default="$2"          # Default value
    local validation="$3"       # Validation type: int, float, or regex pattern
    local error_msg="$4"        # Custom error message
    local min="$5"              # Min value (for int/float)
    local max="$6"              # Max value (for int/float)
    
    local value=""
    local valid=false
    
    # Set default error message if not provided
    error_msg=${error_msg:-"Invalid input. Please try again."}
    
    # Keep prompting until valid input is received
    while [[ "$valid" == false ]]; do
        # Display prompt with default value if applicable
        if [[ -n "$default" ]]; then
            read -p "$prompt (default: $default): " value
            value=${value:-"$default"}
        else
            read -p "$prompt: " value
        fi
        
        # Validate based on type
        if [[ "$validation" == "int" ]]; then
            if ! [[ "$value" =~ ^[0-9]+$ ]]; then
                print_error "$error_msg"
                continue
            fi
            
            # Check range if min/max are specified
            if [[ -n "$min" && "$value" -lt "$min" ]]; then
                print_error "Value must be at least $min"
                continue
            fi
            
            if [[ -n "$max" && "$value" -gt "$max" ]]; then
                print_error "Value must be at most $max"
                continue
            fi
            
        elif [[ "$validation" == "float" ]]; then
            if ! [[ "$value" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
                print_error "$error_msg"
                continue
            fi
            
            # Check range if min/max are specified
            if [[ -n "$min" ]] && (( $(echo "$value < $min" | bc -l) )); then
                print_error "Value must be at least $min"
                continue
            fi
            
            if [[ -n "$max" ]] && (( $(echo "$value > $max" | bc -l) )); then
                print_error "Value must be at most $max"
                continue
            fi
            
        elif [[ -n "$validation" ]]; then
            # Use the validation as a regex pattern
            if ! [[ "$value" =~ $validation ]]; then
                print_error "$error_msg"
                continue
            fi
        fi
        
        # If we got here, input is valid
        valid=true
    done
    
    # Return the validated input
    echo "$value"
}

configure_basic_finetune_settings() {
    # Get model configuration with validation and Mistral-specific defaults
    temperature=$(get_user_input "Enter temperature" "0.7" "float" "Temperature must be a number between 0 and 1" "0" "1")
    top_p=$(get_user_input "Enter top_p" "0.9" "float" "Top-p must be a number between 0 and 1" "0" "1")
    top_k=$(get_user_input "Enter top_k" "50" "int" "Top-k must be a positive integer" "1")
    system_prompt=$(get_user_input "Enter system prompt" "You are Mistral AI's assistant. You are honest, helpful and harmless.")
}

configure_custom_finetune_settings() {
    print_progress "Selecting fine-tuning configuration preset"
    echo ""
    echo "==============================================================="
    echo "                FINE-TUNING CONFIGURATION PRESETS              "
    echo "==============================================================="
    echo ""
    echo "Please select one of the following presets (optimized for RTX 4090 24GB VRAM):"
    echo ""
    echo "1) Fast Training Preset"
    echo "   Best for: Quick iterations and testing fine-tuning results"
    echo "   Characteristics: Aggressive learning rate, fewer epochs, 4-bit quantization"
    echo "   Training time: Fastest (~30-60% faster than balanced preset)"
    echo "   Use when: You want to quickly experiment with different datasets"
    echo "   or fine-tuning approaches, or need results fast"
    echo ""
    echo "2) Balanced Preset (Recommended)"
    echo "   Best for: General purpose fine-tuning with good results"
    echo "   Characteristics: Balanced parameters, 8-bit quantization"
    echo "   Training time: Moderate"
    echo "   Use when: You're looking for reliable, good quality results"
    echo "   without extreme training times"
    echo ""
    echo "3) High-Quality Preset"
    echo "   Best for: Production-ready, high-quality models"
    echo "   Characteristics: Conservative learning rate, more epochs, 8-bit quantization"
    echo "   Training time: Longest (up to 2-3x longer than fast preset)"
    echo "   Use when: Quality is your top priority and you can afford longer"
    echo "   training time for the best possible results"
    echo ""
    
    local preset_choice
    read -p "Select preset (1-3, default: 2): " preset_choice
    preset_choice=${preset_choice:-"2"}
    
    # Default system prompt
    local default_system_prompt="You are Mistral AI's assistant. You are honest, helpful and harmless."
    
    # Set configuration based on selected preset
    if [[ "${preset_choice}" == "1" ]]; then
        # Fast Training Preset
        lora_rank=8
        lora_alpha=16
        lora_dropout=0.03
        learning_rate="5e-4"
        batch_size=6
        grad_acc_steps=2
        num_epochs=2
        warmup_steps=20
        quant_bits=4
        preset_name="Fast Training"
        
    elif [[ "${preset_choice}" == "2" ]]; then
        # Balanced Preset (default values already set)
        preset_name="Balanced"
        
    elif [[ "${preset_choice}" == "3" ]]; then
        # High-Quality Preset
        lora_rank=32
        lora_alpha=64
        lora_dropout=0.07
        learning_rate="1e-4"
        batch_size=4
        grad_acc_steps=8
        num_epochs=5
        warmup_steps=100
        quant_bits=8
        preset_name="High-Quality"
    else
        print_error "Invalid preset selection. Using Balanced preset."
        preset_name="Balanced"
    fi
    
    # Allow system prompt customization
    echo ""
    echo "The current system prompt is:"
    echo "${default_system_prompt}"
    echo ""
    read -p "Would you like to customize the system prompt? (y/n, default: n): " customize_prompt
    customize_prompt=${customize_prompt:-"n"}
    
    if [[ "${customize_prompt}" == "y" || "${customize_prompt}" == "Y" ]]; then
        read -p "Enter your custom system prompt: " custom_system_prompt
        if [[ -n "${custom_system_prompt}" ]]; then
            system_prompt="${custom_system_prompt}"
        fi
    fi
    
    # Generation parameters
    print_progress "Configuring generation parameters"
    
    # Ask if user wants to customize generation parameters
    echo ""
    read -p "Would you like to customize generation parameters? (y/n, default: n): " customize_gen
    customize_gen=${customize_gen:-"n"}
    
    if [[ "${customize_gen}" == "y" || "${customize_gen}" == "Y" ]]; then
        # Temperature
        read -p "Enter temperature (0-1, default: ${temperature}): " custom_temp
        if [[ -n "${custom_temp}" ]] && (( $(echo "${custom_temp} >= 0 && ${custom_temp} <= 1" | bc -l) )); then
            temperature="${custom_temp}"
        fi
        
        # Top_p
        read -p "Enter top_p (0-1, default: ${top_p}): " custom_top_p
        if [[ -n "${custom_top_p}" ]] && (( $(echo "${custom_top_p} >= 0 && ${custom_top_p} <= 1" | bc -l) )); then
            top_p="${custom_top_p}"
        fi
        
        # Top_k
        read -p "Enter top_k (positive integer, default: ${top_k}): " custom_top_k
        if [[ -n "${custom_top_k}" ]] && [[ "${custom_top_k}" =~ ^[0-9]+$ ]] && (( custom_top_k > 0 )); then
            top_k="${custom_top_k}"
        fi
    fi
    
    # Review settings
    echo ""
    echo "==============================================================="
    echo "                FINE-TUNING SETTINGS SUMMARY                   "
    echo "==============================================================="
    echo "Selected Preset: ${preset_name}"
    echo "Base Ollama Model: ${OLLAMA_MODEL}"
    echo "HuggingFace Model: ${HF_MODEL}"
    echo "---------------------------------------------------------------"
    echo "Training Configuration:"
    echo "LoRA Rank (r): ${lora_rank}"
    echo "LoRA Alpha: ${lora_alpha}"
    echo "LoRA Dropout: ${lora_dropout}"
    echo "Learning Rate: ${learning_rate}"
    echo "Batch Size: ${batch_size}"
    echo "Gradient Accumulation Steps: ${grad_acc_steps}"
    echo "Number of Epochs: ${num_epochs}"
    echo "Warmup Steps: ${warmup_steps}"
    echo "Quantization Bits: ${quant_bits}"
    echo "---------------------------------------------------------------"
    echo "Generation Configuration:"
    echo "System Prompt: ${system_prompt}"
    echo "Temperature: ${temperature}"
    echo "Top-p: ${top_p}"
    echo "Top-k: ${top_k}"
    echo "Examples File: ${EXAMPLES_FILE}"
    echo "==============================================================="
    echo ""
    
    read -p "Proceed with these settings? (y/n): " confirm
    if [[ "${confirm}" != "y" && "${confirm}" != "Y" ]]; then
        print_error "Fine-tuning cancelled by user"
        return 1
    fi
}



# ========================================================
# Model Execution Function
# ========================================================

execute_model() {
    log "Starting model execution"
    print_progress "Starting model execution"
    
    # Create web interface first
    create_web_interface
    
    # Get sudo password if not already set
    get_sudo_password
    
    # Start web server
    start_configure_webserver
    
    # List all available models from ollama
    echo "Fetching available models..."
    local all_models=$(ollama list | grep -v "^NAME" | awk '{print $1}')
    
    if [[ -z "${all_models}" ]]; then
        print_error "No models found. Please pull models using 'ollama pull <model_name>' first."
        return 1
    fi
    
    # Separate models into fine-tuned and base models
    local fine_tuned_models=$(echo "$all_models" | grep -E '.*-ft.*')
    local base_models=$(echo "$all_models" | grep -v -E '.*-ft.*')
    
    # Display all models with clear separation
    echo ""
    echo "========================================"
    echo "AVAILABLE MODELS:"
    echo "========================================"
    
    # Display fine-tuned models first if available
    if [[ ! -z "${fine_tuned_models}" ]]; then
        echo ""
        echo "FINE-TUNED MODELS:"
        echo "----------------------------------------"
        readarray -t ft_models_array <<< "$fine_tuned_models"
        for i in "${!ft_models_array[@]}"; do
            # Trim whitespace
            ft_models_array[$i]=$(echo "${ft_models_array[$i]}" | xargs)
            echo "$((i+1)). ${ft_models_array[$i]} [FINE-TUNED]"
        done
        
        local ft_count=${#ft_models_array[@]}
    else
        echo "No fine-tuned models available."
        local ft_count=0
    fi
    
    # Display base models
    echo ""
    echo "BASE MODELS:"
    echo "----------------------------------------"
    readarray -t base_models_array <<< "$base_models"
    for i in "${!base_models_array[@]}"; do
        # Trim whitespace
        base_models_array[$i]=$(echo "${base_models_array[$i]}" | xargs)
        echo "$((ft_count+i+1)). ${base_models_array[$i]}"
    done
    
    # Combined array for selection
    local all_models_array=()
    if [[ $ft_count -gt 0 ]]; then
        all_models_array=("${ft_models_array[@]}" "${base_models_array[@]}")
    else
        all_models_array=("${base_models_array[@]}")
    fi
    
    local total_models=${#all_models_array[@]}
    
    echo ""
    echo "========================================"
    read -p "Enter your choice (1-${total_models}, default: 1): " choice
    choice=${choice:-1}
    
    # Validate input
    if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${total_models}" ]; then
        print_error "Invalid selection. Using default: ${all_models_array[0]}"
        choice=1
    fi
    
    model_name="${all_models_array[$((choice-1))]}"
    
    # Determine if selected model is fine-tuned
    if [[ $choice -le $ft_count ]]; then
        model_type="FINE-TUNED"
    else
        model_type="BASE"
    fi
    
    print_progress "Starting CLI interface for model: ${model_name} [${model_type}]"
    echo ""
    echo "========================================"
    echo "CLI Interface for ${model_name} [${model_type}]"
    echo "Type 'exit' to quit"
    echo "========================================"
    echo ""
    
    local prompt=""
    while true; do
        read -p "> " prompt
        
        if [[ "${prompt}" == "exit" ]]; then
            break
        fi
        
        ollama run "${model_name}" "${prompt}"
        echo ""
    done
    
    print_success "Model execution completed"
    
    # Stop web server when exiting
    stop_webserver
    
    return 0
}

create_custom_finetune_script() {
    local custom_finetune_script="$1"
    print_progress "Creating custom fine-tuning script"
    
    cat > "${custom_finetune_script}" << 'EOF'
#!/usr/bin/env python3
import os
import argparse
import json
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# Add error handling decorator
def error_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"ERROR: {str(e)}")
            import traceback
            print(traceback.format_exc())
            sys.exit(1)
    return wrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLM models with PEFT")
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--ollama_model", type=str, required=True, help="Ollama model name")
    parser.add_argument("--examples_file", type=str, required=True, help="Path to examples.json file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the model")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for models")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for model generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for model generation")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k for model generation")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful and concise assistant specialized in creating HTML, Alpine.js and Tailwind CSS components.", 
                        help="System prompt for the model")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank (r) parameter")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout parameter")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Per device batch size")
    parser.add_argument("--grad_acc_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Warmup steps")
    parser.add_argument("--quant_bits", type=int, default=8, help="Quantization bits: 4, 8, or 16")
    return parser.parse_args()

def prepare_dataset(examples_file):
    with open(examples_file, 'r') as f:
        examples = json.load(f)
    
    # Prepare data for training
    dataset_dict = {
        "input": [example["input"] for example in examples],
        "output": [example["output"] for example in examples]
    }
    
    return Dataset.from_dict(dataset_dict)

def tokenize_function(examples, tokenizer, max_length=512):
    inputs = tokenizer(examples["input"], padding="max_length", truncation=True, max_length=max_length)
    outputs = tokenizer(examples["output"], padding="max_length", truncation=True, max_length=max_length)
    
    # Prepare the labels
    labels = outputs["input_ids"].copy()
    # Set padding tokens to -100 so they are not included in loss computation
    labels = [[-100 if token == tokenizer.pad_token_id else token for token in seq] for seq in labels]
    
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels
    }

def create_ollama_model_file(args, output_dir):
    model_file_content = f"""
FROM {args.ollama_model}

# Set generation parameters
PARAMETER temperature {args.temperature}
PARAMETER top_p {args.top_p}
PARAMETER top_k {args.top_k}

# Set system prompt
SYSTEM {args.system_prompt}
"""
    
    model_file_path = os.path.join(output_dir, "Modelfile")
    with open(model_file_path, 'w') as f:
        f.write(model_file_content.strip())
    
    return model_file_path

@error_handler
def main():
    args = parse_args()
    
    print(f"Loading model: {args.model_name}")
    print(f"Using preset configuration with LoRA rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"Training settings: lr={args.learning_rate}, batch_size={args.batch_size}, gradient_accumulation={args.grad_acc_steps}, epochs={args.num_epochs}")
    
    # Set up comprehensive error handling for dependencies
    # Check for protobuf installation
    try:
        import google.protobuf
        print(f"Protobuf is properly installed (version: {google.protobuf.__version__})")
    except ImportError as e:
        print(f"WARNING: Google protobuf error: {str(e)}. Installing it now...")
        import subprocess
        try:
            # Try a specific version known to work with SentencePiece
            subprocess.run(["pip", "install", "--force-reinstall", "protobuf==3.20.3"], check=True)
            print("Installed protobuf 3.20.3. Trying to import again...")
            import google.protobuf
            print(f"Protobuf is now installed (version: {google.protobuf.__version__})")
        except Exception as install_error:
            print(f"ERROR installing protobuf: {str(install_error)}")
            print("Trying to continue anyway, but this may cause errors...")
    
    # Check for SentencePiece installation
    try:
        import sentencepiece
        print(f"SentencePiece is properly installed (version: {sentencepiece.__version__})")
    except ImportError as e:
        print(f"WARNING: SentencePiece error: {str(e)}. Installing it now...")
        import subprocess
        try:
            subprocess.run(["pip", "install", "--force-reinstall", "sentencepiece>=0.1.99"], check=True)
            print("Installed SentencePiece. Trying to import again...")
            import sentencepiece
            print(f"SentencePiece is now installed (version: {sentencepiece.__version__})")
        except Exception as install_error:
            print(f"ERROR installing SentencePiece: {str(install_error)}")
            raise ImportError("Critical dependency SentencePiece could not be installed. Cannot continue.") 
    
    # Create BitsAndBytesConfig for quantization
    from transformers import BitsAndBytesConfig
    
    # Configure quantization based on user settings
    quantization_config = None
    if args.quant_bits == 4:
        print("Using 4-bit quantization as specified")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif args.quant_bits == 8:
        print("Using 8-bit quantization as specified")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
    else:
        # 16-bit (no quantization)
        print("Using 16-bit precision (no quantization) as specified")
        quantization_config = None
    
    # Load the model and tokenizer with updated configuration
    model_load_params = {
        "device_map": "auto",
        "cache_dir": args.cache_dir,
        "trust_remote_code": True
    }
    
    # Add quantization config if we're using quantization
    if quantization_config:
        model_load_params["quantization_config"] = quantization_config
    else:
        model_load_params["torch_dtype"] = torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_load_params)
    
    # First try with fast tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Fast tokenizer failed with error: {str(e)}")
        print("Falling back to slow tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA - dynamically determine target modules based on model architecture
    target_modules = []
    
    # Check model architecture to set appropriate target modules
    for name, _ in model.named_modules():
        if any(module_type in name for module_type in ["q_proj", "v_proj", "k_proj", "o_proj"]):
            if "q_proj" not in target_modules and "q_proj" in name:
                target_modules.append("q_proj")
            if "v_proj" not in target_modules and "v_proj" in name:
                target_modules.append("v_proj")
            if "k_proj" not in target_modules and "k_proj" in name:
                target_modules.append("k_proj")
            if "o_proj" not in target_modules and "o_proj" in name:
                target_modules.append("o_proj")
        
        if any(module_type in name for module_type in ["gate_proj", "up_proj", "down_proj"]):
            if "gate_proj" not in target_modules and "gate_proj" in name:
                target_modules.append("gate_proj")
            if "up_proj" not in target_modules and "up_proj" in name:
                target_modules.append("up_proj")
            if "down_proj" not in target_modules and "down_proj" in name:
                target_modules.append("down_proj")
    
    # If no modules were found, use a default set for common architectures
    if not target_modules:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    print(f"Using LoRA target modules: {target_modules}")
    
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules
    )
    
    model = get_peft_model(model, lora_config)
    
    # Load and prepare dataset
    dataset = prepare_dataset(args.examples_file)
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_steps=args.warmup_steps,
        optim="paged_adamw_8bit",
        report_to=None,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        load_best_model_at_end=False,
        save_total_limit=3,  # Keep only the 3 best checkpoints
        resume_from_checkpoint=True 
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=None 
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the model
    print(f"Saving fine-tuned model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Create Ollama model file
    model_file_path = create_ollama_model_file(args, args.output_dir)
    print(f"Created Ollama model file at {model_file_path}")
    
    print("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()
EOF

    chmod +x "${custom_finetune_script}"
    print_success "Custom fine-tuning script created successfully"
}

create_finetune_wrapper_script() {
    local wrapper_script="$1"
    print_progress "Creating fine-tuning wrapper script"
    
    cat > "${wrapper_script}" << 'EOWRAPPER'
#!/usr/bin/env python3
import sys
import subprocess
import os
import importlib.util

# List of critical dependencies
DEPENDENCIES = [
    ("google.protobuf", "protobuf==3.20.3"),
    ("sentencepiece", "sentencepiece>=0.1.99"),
    ("transformers", "transformers"),
    ("peft", "peft"),
    ("torch", "torch"),
    ("datasets", "datasets")
]

# Check and install dependencies
for module_name, package_name in DEPENDENCIES:
    try:
        # Try to import the module
        importlib.import_module(module_name)
        print(f"✓ {module_name} is already installed")
    except ImportError:
        print(f"✗ {module_name} is not installed. Installing {package_name}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--force-reinstall", package_name], check=True)
            print(f"✓ Successfully installed {package_name}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package_name}")
            sys.exit(1)

# Now run the actual fine-tuning script with all arguments
finetune_script = sys.argv[1]
args = sys.argv[2:]

print(f"\nRunning fine-tuning script: {finetune_script}")
print(f"With arguments: {' '.join(args)}")

# Execute the fine-tuning script
try:
    result = subprocess.run([sys.executable, finetune_script] + args, check=True)
    sys.exit(result.returncode)
except subprocess.CalledProcessError as e:
    print(f"Fine-tuning failed with exit code: {e.returncode}")
    sys.exit(e.returncode)
EOWRAPPER

    chmod +x "${wrapper_script}"
    print_success "Fine-tuning wrapper script created successfully"
}

# ========================================================
# Web Server Function
# ========================================================

# Fix for the start_configure_webserver function

start_configure_webserver() {
    log "Starting and configuring web server"
    print_progress "Starting and configuring web server"

    # Ensure script is run with sudo
    if [[ $EUID -ne 0 ]]; then
        # If we have sudo password, use it
        if [[ -n "${SUDO_PASSWORD}" ]]; then
            # Use sudo to execute commands directly instead of recursively calling the function
            echo "${SUDO_PASSWORD}" | sudo -S bash -c "
                # Ollama service configuration
                OLLAMA_SERVICE_FILE=\"/etc/systemd/system/ollama.service\"
                if [ -f \"\$OLLAMA_SERVICE_FILE\" ]; then
                    if ! grep -q \"Environment=\\\"OLLAMA_HOST=0.0.0.0:11434\\\"\" \"\$OLLAMA_SERVICE_FILE\"; then
                        sed -i '/\\[Service\\]/a Environment=\"OLLAMA_HOST=0.0.0.0:11434\"' \"\$OLLAMA_SERVICE_FILE\"
                        echo \"Added OLLAMA_HOST configuration to Ollama service file\"
                    fi
                    # Restart Ollama service
                    systemctl daemon-reload
                    systemctl restart ollama
                fi
                
                # Configure UFW if active
                if command -v ufw &> /dev/null && ufw status | grep -q \"active\"; then
                    ufw allow 11434/tcp
                fi
            "
            
            # Check if the sudo command was successful
            if [[ $? -ne 0 ]]; then
                print_error "Failed to configure Ollama service with sudo"
                # Continue anyway to start the web server
            fi
        else
            print_error "Web server configuration requires sudo privileges for optimal setup"
            print_progress "Continuing with limited configuration"
            # Continue anyway to start the web server
        fi
    else
        # We are already running as root, so execute directly
        
        # Ollama service configuration
        OLLAMA_SERVICE_FILE="/etc/systemd/system/ollama.service"
        if [ -f "$OLLAMA_SERVICE_FILE" ]; then
            if ! grep -q "Environment=\"OLLAMA_HOST=0.0.0.0:11434\"" "$OLLAMA_SERVICE_FILE"; then
                sed -i '/\[Service\]/a Environment="OLLAMA_HOST=0.0.0.0:11434"' "$OLLAMA_SERVICE_FILE"
                echo "Added OLLAMA_HOST configuration to Ollama service file"
            fi
            # Restart Ollama service
            systemctl daemon-reload
            systemctl restart ollama
        fi
        
        # Configure UFW if active
        if command -v ufw &> /dev/null && ufw status | grep -q "active"; then
            ufw allow 11434/tcp
        fi
    fi

    # Create a directory for server logs (this doesn't require sudo)
    mkdir -p "${SCRIPT_DIR}/logs"
    SERVER_LOG="${SCRIPT_DIR}/logs/python_server.log"

    # Create the server.py file if it doesn't exist
    create_custom_server_script

    # Start Python Server with Error Handling
    log "Starting Python server"
    # Kill any existing Python server on port 8000
    pkill -f "python.*server.py" || true
    
    # Start the custom Python server
    "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/server.py" > "${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!

    # Store the PID for later termination
    echo "${SERVER_PID}" > "${SCRIPT_DIR}/web_server.pid"

    # Verify server started
    sleep 2
    if ps -p $SERVER_PID > /dev/null; then
        log "Python server started successfully with PID $SERVER_PID on port 8000"
        print_success "Web server is running on port 8000"
        
        # Print IP addresses for access
        echo "You can access the web interface at:"
        ip_addresses=$(hostname -I)
        for ip in $ip_addresses; do
            echo "http://${ip}:8000"
        done
        
        # Log service statuses
        log "Ollama service status:"
        systemctl is-active ollama
        log "UFW status for port 11434 (if applicable):"
        if command -v ufw &> /dev/null; then
            ufw status | grep 11434 || echo "No specific rule for port 11434"
        fi
        
        return 0
    else
        log "ERROR: Python server failed to start"
        log "Server log contents (if any):"
        cat "${SERVER_LOG}"
        return 1
    fi
}

# Function to create the custom server script
create_custom_server_script() {
    print_progress "Creating custom server script"
    
    cat > "${SCRIPT_DIR}/server.py" << 'EOF'
#!/usr/bin/env python3
import http.server
import socketserver
import json
import os
import sys
from urllib.parse import urlparse, parse_qs

# Configuration
PORT = 8001
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
EXAMPLES_FILE = os.path.join(DATA_DIR, "examples.json")

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize examples.json if it doesn't exist
if not os.path.exists(EXAMPLES_FILE):
    with open(EXAMPLES_FILE, 'w') as f:
        json.dump([{"input": "Create a single html tag", "output":"<html>example output</html>"}], f)
    print(f"Created initial examples.json at {EXAMPLES_FILE}")

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_url = urlparse(self.path)
        
        # Handle request for examples.json
        if parsed_url.path == "/data/examples.json":
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            try:
                with open(EXAMPLES_FILE, 'r') as f:
                    self.wfile.write(f.read().encode())
            except Exception as e:
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            # Default behavior for static files
            return http.server.SimpleHTTPRequestHandler.do_GET(self)
    
    def do_POST(self):
        parsed_url = urlparse(self.path)
        
        # Handle POST to examples.json
        if parsed_url.path == "/data/examples.json":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            
            try:
                # Parse the JSON data
                examples_data = json.loads(post_data)
                
                # Write to the examples.json file
                with open(EXAMPLES_FILE, 'w') as f:
                    json.dump(examples_data, f, indent=2)
                
                # Send success response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success"}).encode())
                
                print(f"Updated examples.json with new data. Total examples: {len(examples_data)}")
            except Exception as e:
                # Send error response
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
                print(f"Error updating examples.json: {str(e)}")
        else:
            # Method not allowed for other paths
            self.send_response(405)
            self.end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def run_server():
    handler = CustomHTTPRequestHandler
    
    # Use ThreadingTCPServer to handle multiple requests
    with socketserver.ThreadingTCPServer(("", PORT), handler) as httpd:
        print(f"Server running at http://localhost:{PORT}")
        print(f"Data directory: {DATA_DIR}")
        print("Press Ctrl+C to stop the server...")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.shutdown()

if __name__ == "__main__":
    run_server()
EOF

    chmod +x "${SCRIPT_DIR}/server.py"
    print_success "Custom server script created successfully"
}

# Stop the web server
stop_webserver() {
    log "Stopping web server"
    print_progress "Stopping web server"
    
    if [[ -f "${SCRIPT_DIR}/web_server.pid" ]]; then
        local pid=$(cat "${SCRIPT_DIR}/web_server.pid")
        if ps -p $pid > /dev/null; then
            print_progress "Terminating web server process (PID: $pid)"
            kill $pid
            rm "${SCRIPT_DIR}/web_server.pid"
            print_success "Web server stopped successfully"
        else
            print_progress "Web server process not running (PID: $pid)"
            rm "${SCRIPT_DIR}/web_server.pid"
        fi
    else
        print_progress "No web server PID file found"
    fi
}

# Create index.html in the current directory
create_web_interface() {
    log "Creating web interface files"
    print_progress "Creating web interface files"
    
    # Create index.html from the template with Tailwind CSS styling
    cat > "${SCRIPT_DIR}/index.html" << 'EOL'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ollama Multi-Model Chat</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        primary: {
                            DEFAULT: '#4A6CFF',
                            hover: '#6A8CFF'
                        },
                        dark: {
                            bg: {
                                primary: '#0A0A0F',
                                secondary: '#121218',
                                tertiary: '#1C1C24'
                            },
                            text: {
                                primary: '#E0E0E8',
                                secondary: '#A0A0B0'
                            },
                            border: '#2C2C3A'
                        }
                    }
                }
            }
        }
    </script>
</head>
<body class="bg-dark-bg-primary text-dark-text-primary font-sans h-screen overflow-hidden dark" x-data="app">
    <div class="flex h-screen">
        <!-- Top control bar -->
        <div class="w-full h-screen flex flex-col">
            <!-- Header with controls -->
            <div class="bg-dark-bg-secondary border-b border-dark-border p-4 flex flex-wrap gap-4 items-center">
                <div class="text-2xl font-bold text-primary mr-4">Ollama</div>
                
                <div class="flex items-center gap-2">
                    <input type="text" id="server-url" x-model="baseUrl" placeholder="Ollama Server URL" 
                        class="bg-dark-bg-tertiary border border-dark-border text-dark-text-primary p-2 rounded text-sm w-64">
                    <button @click="fetchModels()" class="bg-primary hover:bg-primary-hover text-white px-4 py-2 rounded flex items-center gap-2 transition-colors">
                        <i class="fas fa-plug"></i>
                        Connect
                    </button>
                </div>
                
                <div class="flex-grow"></div>
                
                <!-- Model selection area with checkboxes -->
                <div class="flex items-center gap-2">
                    <span class="text-primary font-semibold">Models (Max 6):</span>
                    <div id="model-checkboxes" class="flex flex-wrap gap-3 bg-dark-bg-tertiary border border-dark-border p-2 rounded">
                        <template x-if="loadingModels">
                            <div class="flex items-center">
                                <div class="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin mr-2"></div>
                                <span>Loading models...</span>
                            </div>
                        </template>
                        <template x-if="!loadingModels && models.length === 0">
                            <div class="text-dark-text-secondary italic">Connect to load models...</div>
                        </template>
                        <template x-for="model in models" :key="model.name">
                            <div class="flex items-center">
                                <input type="checkbox" :id="'model-' + model.name" :value="model.name" 
                                    @change="updateModelSelection($event)" 
                                    class="mr-2 model-checkbox w-4 h-4 accent-primary">
                                <label :for="'model-' + model.name" :title="'Size: ' + model.size" 
                                    class="text-dark-text-primary cursor-pointer hover:text-primary transition-colors" 
                                    x-text="model.name"></label>
                            </div>
                        </template>
                    </div>
                </div>
            </div>
            
            <!-- Message input area - Reorganized to horizontal layout -->
            <div class="bg-dark-bg-secondary p-4 border-b border-dark-border">
                <!-- Unified input row -->
                <div class="flex gap-3 mb-4">
                    <!-- Preamble input - doubled height from h-24 to h-48 -->
                    <div class="w-1/4">
                        <div class="flex items-center text-primary font-semibold mb-1">
                            <i class="fas fa-chevron-up mr-2"></i>
                            <span>Preamble</span>
                        </div>
                        <textarea id="preamble-input" x-model="preamble" placeholder="Enter preamble..." rows="6" 
                            :disabled="selectedModels.length === 0"
                            class="w-full bg-dark-bg-tertiary text-dark-text-primary border border-dark-border p-3 rounded-lg resize-none text-base leading-relaxed shadow-inner focus:ring-2 focus:ring-primary focus:outline-none transition-all h-64"></textarea>
                    </div>
                    
                    <!-- Main message input - doubled height from h-24 to h-48 -->
                    <div class="w-1/2 relative">
                        <div class="flex items-center text-primary font-semibold mb-1">
                            <i class="fas fa-comment mr-2"></i>
                            <span>Message</span>
                        </div>
                        <textarea id="user-input" x-model="userMessage" placeholder="Type your message..." rows="6" 
                            :disabled="selectedModels.length === 0"
                            @keydown.enter.prevent="$event.shiftKey ? null : sendMessage()"
                            class="w-full bg-dark-bg-tertiary text-dark-text-primary border border-dark-border p-3 rounded-lg resize-none text-base leading-relaxed shadow-inner focus:ring-2 focus:ring-primary focus:outline-none transition-all h-64"></textarea>
                        <div class="absolute bottom-3 right-3 text-dark-text-secondary text-xs opacity-70">Shift+Enter for new line, Enter to send</div>
                    </div>
                    
                    <!-- Signoff input - doubled height from h-24 to h-48 -->
                    <div class="w-1/4">
                        <div class="flex items-center text-primary font-semibold mb-1">
                            <i class="fas fa-chevron-down mr-2"></i>
                            <span>Signoff</span>
                        </div>
                        <textarea id="signoff-input" x-model="signoff" placeholder="Enter signoff..." rows="6" 
                            :disabled="selectedModels.length === 0"
                            class="w-full bg-dark-bg-tertiary text-dark-text-primary border border-dark-border p-3 rounded-lg resize-none text-base leading-relaxed shadow-inner focus:ring-2 focus:ring-primary focus:outline-none transition-all h-64"></textarea>
                    </div>
                </div>
                
                <div class="flex gap-3">
                    <!-- Last submitted message display - doubled height from max-h-32 to max-h-64 -->
                    <div class="flex-grow">
                        <div class="text-primary font-semibold mb-1 flex items-center">
                            <i class="fas fa-history mr-2"></i>
                            Last Message:
                        </div>
                        <div id="last-message" 
                            :class="{'italic text-dark-text-secondary': lastMessage === '', 'text-dark-text-primary': lastMessage !== ''}"
                            class="bg-dark-bg-tertiary border border-dark-border p-3 rounded-lg max-h-64 overflow-y-auto shadow-inner">
                            <template x-if="lastMessage === ''">
                                <span>No message sent yet</span>
                            </template>
                            <template x-if="lastMessage !== ''">
                                <div x-html="formatLastMessage()"></div>
                            </template>
                        </div>
                    </div>
                    
                    <!-- Send button -->
                    <div class="flex items-end">
                        <button @click="sendMessage()" 
                            :disabled="userMessage.trim() === '' || selectedModels.length === 0 || sendingMessage"
                            class="bg-primary hover:bg-primary-hover text-white px-6 py-3 rounded-lg flex items-center justify-center gap-2 transition-colors shadow-lg hover:shadow-xl disabled:opacity-50">
                            <i class="fas fa-paper-plane"></i>
                            Send
                        </button>
                    </div>
                </div>
            </div>

            <!-- Main content area with model previews -->
            <div class="flex-grow bg-dark-bg-primary overflow-hidden">
                <div id="preview-container" 
                    :class="getGridClass()"
                    class="h-full p-4 grid gap-4">
                    <template x-if="selectedModels.length === 0">
                        <div class="col-span-3 row-span-2 flex items-center justify-center text-dark-text-secondary text-lg">
                            <div class="text-center p-8 rounded-lg bg-dark-bg-secondary border border-dark-border shadow-lg">
                                <i class="fas fa-robot text-6xl mb-4 text-primary opacity-50"></i>
                                <p>Select one or more models to see responses here</p>
                            </div>
                        </div>
                    </template>
                    
                    <template x-for="model in selectedModels" :key="model">
                        <div :id="'preview-' + model" class="model-preview-panel h-full bg-dark-bg-secondary border border-dark-border rounded-lg flex flex-col shadow-md" x-data="{ viewMode: 'preview' }">
                            <div class="flex justify-between items-center p-3 border-b border-dark-border">
                                <div class="flex items-center">
                                    <div class="model-name font-semibold text-primary" x-text="model"></div>
                                    <div class="response-time ml-2 text-xs text-dark-text-secondary hidden bg-dark-bg-tertiary px-2 py-1 rounded-full">
                                        <i class="fas fa-clock mr-1"></i>
                                        <span x-text="modelResponseTimes[model] || '0ms'"></span>
                                    </div>
                                </div>
                                <div class="flex gap-2">
                                    <button @click="viewMode = 'code'" 
                                        :class="{'opacity-50': viewMode !== 'code'}"
                                        class="bg-dark-bg-tertiary hover:bg-dark-border px-3 py-1 rounded-lg text-sm transition-colors">
                                        <i class="fas fa-code"></i> Code
                                    </button>
                                    <button @click="viewMode = 'preview'" 
                                        :class="{'opacity-50': viewMode !== 'preview'}"
                                        class="bg-dark-bg-tertiary hover:bg-dark-border px-3 py-1 rounded-lg text-sm transition-colors">
                                        <i class="fas fa-eye"></i> Preview
                                    </button>
                                    <button @click="saveExample(model)" 
                                        class="save-example-btn bg-dark-bg-tertiary hover:bg-dark-border px-2 py-1 rounded-lg text-sm transition-colors ml-1" 
                                        title="Save as example">
                                        <i class="fas fa-save"></i>
                                    </button>
                                    <button @click="copyToClipboard(model)" 
                                        class="copy-btn bg-dark-bg-tertiary hover:bg-dark-border px-2 py-1 rounded-lg text-sm transition-colors ml-1" 
                                        title="Copy to clipboard">
                                        <i class="fas fa-copy"></i>
                                    </button>
                                </div>
                            </div>
                            <div class="preview-container flex-grow overflow-hidden relative">
                                <div class="code-view absolute inset-0 overflow-auto p-3 whitespace-pre-wrap break-words text-dark-text-secondary font-mono text-sm"
                                    :class="{'hidden': viewMode !== 'code'}"
                                    x-html="formatCodeView(model)">
                                </div>
                                <div class="content-view absolute inset-0"
                                    :class="{'hidden': viewMode !== 'preview'}"
                                    x-html="formatContentView(model)">
                                </div>
                                <div class="spinner absolute inset-0 flex items-center justify-center bg-dark-bg-secondary bg-opacity-80"
                                    :class="{'hidden': !modelLoadingStates[model]}">
                                    <div class="flex flex-col items-center">
                                        <div class="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mb-4"></div>
                                        <div class="text-dark-text-primary">Generating response...</div>
                                    </div>
                                </div>
                            </div>
                            <!-- Remix input and button area -->
                            <div class="flex items-center p-3 border-t border-dark-border">
                                <input type="text" x-model="remixInstructions[model]" 
                                    @keydown.enter.prevent="remixResponse(model)"
                                    class="remix-input flex-grow bg-dark-bg-tertiary border border-dark-border text-dark-text-primary p-2 rounded-lg text-sm mr-2" 
                                    placeholder="Enter remix instructions... (Press Enter to submit)">
                                <button @click="remixResponse(model)" class="remix-btn bg-dark-bg-tertiary hover:bg-dark-border px-3 py-2 rounded-lg text-sm transition-colors flex items-center gap-1">
                                    <i class="fas fa-sync-alt"></i> Remix
                                </button>
                            </div>
                        </div>
                    </template>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.addEventListener('alpine:init', () => {
            Alpine.data('app', () => ({
                baseUrl: 'http://localhost:11434',
                models: [],
                selectedModels: [],
                loadingModels: false,
                userMessage: '',
                preamble: localStorage.getItem('ollamaPreamble') || '',
                signoff: localStorage.getItem('ollamaSignoff') || '',
                lastMessage: '',
                modelResponses: {},
                modelLoadingStates: {},
                sendingMessage: false,
                modelResponseTimes: {},
                remixInstructions: {},
                EXAMPLES_FILE_PATH: '/data/examples.json',
                
                init() {
                    // Initialize from local storage
                    this.preamble = localStorage.getItem('ollamaPreamble') || '';
                    this.signoff = localStorage.getItem('ollamaSignoff') || '';
                    this.fetchModels();
                },
                
                async fetchModels() {
                    try {
                        this.loadingModels = true;
                        this.models = [];
                        
                        const response = await fetch(`${this.baseUrl}/api/tags`, {
                            method: 'GET',
                            mode: 'cors',
                            headers: {
                                'Content-Type': 'application/json'
                            }
                        });
                        
                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }
                        
                        const data = await response.json();
                        
                        if (data.models && data.models.length > 0) {
                            this.models = data.models;
                            
                            // Auto-select all models up to maximum of 6
                            this.selectedModels = [];
                            const modelsToSelect = this.models.slice(0, 6);
                            this.selectedModels = modelsToSelect.map(model => model.name);
                            
                            // Update checkboxes to match selections
                            setTimeout(() => {
                                modelsToSelect.forEach(model => {
                                    const checkbox = document.getElementById(`model-${model.name}`);
                                    if (checkbox) checkbox.checked = true;
                                });
                            }, 100);
                        } else {
                            this.showNotification("No models found. Try pulling some models with 'ollama pull'.", "error");
                        }
                    } catch (error) {
                        console.error('Error fetching models:', error);
                        this.showNotification(`Error connecting to Ollama server: ${error.message}`, "error");
                    } finally {
                        this.loadingModels = false;
                    }
                },
                
                updateModelSelection(event) {
                    const model = event.target.value;
                    const isChecked = event.target.checked;
                    
                    if (isChecked) {
                        // Limit to 6 models maximum
                        if (this.selectedModels.length >= 6) {
                            event.target.checked = false;
                            this.showNotification("You can select up to 6 models maximum.", "info");
                            return;
                        }
                        this.selectedModels.push(model);
                    } else {
                        this.selectedModels = this.selectedModels.filter(m => m !== model);
                    }
                },
                
                getGridClass() {
                    if (this.selectedModels.length === 1) {
                        return 'grid-cols-1 grid-rows-1';
                    } else if (this.selectedModels.length === 2) {
                        return 'grid-cols-2 grid-rows-1';
                    } else if (this.selectedModels.length === 3) {
                        return 'grid-cols-3 grid-rows-1';
                    } else if (this.selectedModels.length === 4) {
                        return 'grid-cols-2 grid-rows-2';
                    } else if (this.selectedModels.length === 5 || this.selectedModels.length === 6) {
                        return 'grid-cols-3 grid-rows-2';
                    }
                    return '';
                },
                
                formatLastMessage() {
                    return this.lastMessage.replace(/\n/g, '<br>');
                },
                
                async sendMessage() {
                    const messageText = this.userMessage.trim();
                    if (!messageText || this.selectedModels.length === 0) return;
                    
                    // Combine preamble, message, and signoff
                    let fullMessage = messageText;
                    if (this.preamble.trim()) {
                        fullMessage = this.preamble.trim() + '\n\n' + fullMessage;
                    }
                    if (this.signoff.trim()) {
                        fullMessage = fullMessage + '\n\n' + this.signoff.trim();
                    }
                    
                    // Save preamble and signoff to localStorage
                    localStorage.setItem('ollamaPreamble', this.preamble);
                    localStorage.setItem('ollamaSignoff', this.signoff);
                    
                    // Update last message
                    this.lastMessage = fullMessage;
                    
                    // Add a subtle animation to the last message display
                    const lastMessageElement = document.getElementById('last-message');
                    lastMessageElement.style.transition = 'background-color 0.3s';
                    lastMessageElement.style.backgroundColor = '#2C2C3A';
                    setTimeout(() => {
                        lastMessageElement.style.backgroundColor = '';
                    }, 500);
                    
                    // Clear input field
                    this.userMessage = '';
                    
                    // Disable send button while processing
                    this.sendingMessage = true;
                    
                    // Reset responses
                    this.modelResponses = {};
                    
                    // Send to each selected model
                    const promises = this.selectedModels.map(model => this.sendToModel(model, fullMessage));
                    
                    // Re-enable send button when all requests complete
                    Promise.all(promises).finally(() => {
                        this.sendingMessage = false;
                    });
                },
                
                async sendToModel(model, message) {
                    // Show loading state
                    this.modelLoadingStates[model] = true;
                    this.modelResponses[model] = '';
                    
                    // Mark response time as hidden
                    const panel = document.getElementById(`preview-${model}`);
                    if (panel) {
                        const responseTimeElement = panel.querySelector('.response-time');
                        if (responseTimeElement) {
                            responseTimeElement.classList.add('hidden');
                        }
                    }
                    
                    // Prepare request
                    const requestOptions = {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            model: model,
                            prompt: message,
                            stream: false
                        })
                    };
                    
                    const startTime = performance.now();
                    
                    try {
                        const response = await fetch(`${this.baseUrl}/api/generate`, requestOptions);
                        
                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }
                        
                        const data = await response.json();
                        const endTime = performance.now();
                        const responseTime = Math.round(endTime - startTime);
                        
                        // Update response time
                        this.modelResponseTimes[model] = `${responseTime}ms`;
                        
                        // Show response time
                        if (panel) {
                            const responseTimeElement = panel.querySelector('.response-time');
                            if (responseTimeElement) {
                                responseTimeElement.classList.remove('hidden');
                            }
                        }
                        
                        // Store response
                        this.modelResponses[model] = data.response;
                        
                        // Automatically switch to preview mode when response is received
                        if (panel && panel.__x) {
                            panel.__x.$data.viewMode = 'preview';
                        }
                    } catch (error) {
                        console.error(`Error generating response for ${model}:`, error);
                        this.modelResponses[model] = `Error: ${error.message}`;
                        this.showNotification(`Error with ${model}: ${error.message}`, "error");
                    } finally {
                        // Hide loading state
                        this.modelLoadingStates[model] = false;
                    }
                },
                
                // Replace the existing remixResponse function with this updated version
                async remixResponse(model) {
                    const remixInstructions = this.remixInstructions[model]?.trim();
                    
                    if (!remixInstructions) {
                        this.showNotification("Please enter remix instructions", "error");
                        return;
                    }
                    
                    // Get the original response from the model where the remix was requested
                    const originalResponse = this.modelResponses[model];
                    if (!originalResponse) {
                        this.showNotification(`No response to remix from ${model}`, "error");
                        return;
                    }
                    
                    // Extract HTML if available, or use full response
                    const codeToRemix = this.extractHTML(originalResponse) || originalResponse;
                    
                    // Apply the remix to all selected models using the source model's code
                    const promises = this.selectedModels.map(currentModel => {
                        // Build the remix prompt - using the source model's code for all models
                        let remixPrompt = "Review the following code:" + '\n\n' + codeToRemix + '\n\n' + 
                                        "Modify this according to the following instructions: " + remixInstructions;
                        
                        // Add signoff if it exists
                        if (this.signoff.trim()) {
                            remixPrompt = remixPrompt + '\n\n' + this.signoff.trim();
                        }
                        
                        // Send to each model
                        return this.sendToModel(currentModel, remixPrompt);
                    });
                    
                    // Update last message
                    this.lastMessage = "Remix instructions: " + remixInstructions + 
                                    "\n\nSource code from: " + model;
                    
                    // Wait for all remix requests to complete
                    await Promise.all(promises);
                    
                    // Show notification
                    this.showNotification(`Remix applied to all models using ${model}'s code`, "success");
                    
                    // Clear the remix instruction only from the triggering model
                    this.remixInstructions[model] = '';
                },
                                
                formatCodeView(model) {
                    if (!this.modelResponses[model]) {
                        return '<p class="italic text-dark-text-secondary">No response yet</p>';
                    }
                    
                    // Show extracted HTML if available, or full response
                    const extractedHTML = this.extractHTML(this.modelResponses[model]);
                    const content = extractedHTML || this.modelResponses[model];
                    
                    // Apply code formatting - this is the key improvement
                    let formattedContent = content;
                    
                    if (extractedHTML) {
                        // Format HTML with proper indentation for better readability
                        formattedContent = this.formatHTML(extractedHTML);
                    } else {
                        // For non-HTML content, preserve whitespace and line breaks
                        formattedContent = content;
                    }
                    
                    // Escape the content for display and ensure whitespace is preserved
                    return `<pre class="whitespace-pre overflow-x-auto p-3">${this.escapeHtml(formattedContent)}</pre>`;
                },
                
                formatContentView(model) {
                    if (!this.modelResponses[model]) {
                        return '<p class="italic text-dark-text-secondary p-3">Content preview will appear here</p>';
                    }
                    
                    try {
                        // Debug information can be enabled for troubleshooting
                        if (this.debug) {
                            this.debugHtmlRendering(model);
                        }
                        
                        const response = this.modelResponses[model];
                        
                        // Check if we have HTML content wrapped in markdown code blocks
                        const htmlCodeBlockRegex = /```html\s*(<!DOCTYPE html>[\s\S]*?<\/html>)\s*```/;
                        const htmlMatch = response.match(htmlCodeBlockRegex);
                        
                        let htmlContent = null;
                        
                        if (htmlMatch && htmlMatch[1]) {
                            // Extract the HTML content from inside the code block
                            htmlContent = htmlMatch[1];
                        } else {
                            // Try the regular HTML extraction for non-code-block HTML
                            const docTypeMatch = response.match(/<!DOCTYPE html>[\s\S]*?<\/html>/);
                            if (docTypeMatch) {
                                htmlContent = docTypeMatch[0];
                            }
                        }
                        
                        if (htmlContent) {
                            // Process the HTML to ensure it's valid
                            const processedHTML = this.processHTMLContent(htmlContent);
                            
                            if (processedHTML) {
                                // Create iframe with the processed HTML
                                return `<div class="w-full h-full absolute inset-0">
                                    <iframe sandbox="allow-scripts" class="w-full h-full border-0" 
                                        srcdoc="${this.escapeHtml(processedHTML)}"></iframe>
                                </div>`;
                            }
                        }
                        
                        // If we couldn't extract or process HTML, fall back to text formatting
                        // Format markdown-style code blocks
                        let formattedText = response.replace(/```([a-z]*)\n([\s\S]*?)```/g, 
                            '<pre class="bg-dark-bg-primary rounded-lg p-3 my-3 overflow-x-auto font-mono"><code class="text-dark-text-secondary text-sm">$2</code></pre>');
                        
                        // Format inline code
                        formattedText = formattedText.replace(/`([^`]+)`/g, 
                            '<code class="bg-dark-bg-primary px-2 py-1 rounded text-dark-text-secondary text-sm">$1</code>');
                        
                        // Format line breaks
                        formattedText = formattedText.replace(/\n/g, '<br>');
                        
                        return `<div class="p-4">${formattedText}</div>`;
                    } catch (error) {
                        console.error(`Error in formatContentView for model ${model}:`, error);
                        // Provide a user-friendly error message
                        return `<div class="p-4 text-red-500">
                            <p class="font-bold">Error rendering content</p>
                            <p>There was an error processing the content. You can still view the raw response in the code tab.</p>
                            <p class="mt-2 text-xs">${this.escapeHtml(error.message)}</p>
                        </div>`;
                    }
                },

                // Simple debug utility for HTML rendering issues
                debugHtmlRendering(model) {
                    const response = this.modelResponses[model];
                    if (!response) {
                        console.log(`No response for model ${model}`);
                        return;
                    }
                    
                    console.group(`HTML Rendering Debug: ${model}`);
                    
                    // Check if there's HTML in the response
                    const hasHTML = response.includes('<!DOCTYPE html>');
                    console.log(`Contains DOCTYPE: ${hasHTML}`);
                    
                    if (hasHTML) {
                        // Check for key elements
                        console.log(`Contains <head>: ${response.includes('<head>')}`);
                        console.log(`Contains </head>: ${response.includes('</head>')}`);
                        console.log(`Contains <body>: ${response.includes('<body')}`);
                        console.log(`Contains </body>: ${response.includes('</body>')}`);
                        console.log(`Contains tailwind.config: ${response.includes('tailwind.config')}`);
                        
                        // Check script tags
                        const openScriptTags = (response.match(/<script/g) || []).length;
                        const closeScriptTags = (response.match(/<\/script>/g) || []).length;
                        console.log(`Script tags: ${openScriptTags} opening, ${closeScriptTags} closing`);
                        
                        if (openScriptTags !== closeScriptTags) {
                            console.warn('Mismatched script tags detected!');
                        }
                    }
                    
                    console.groupEnd();
                },

                extractHTML(text) {
                    if (!text) return null;
                    
                    try {
                        // First check for HTML content that might be inside markdown code blocks
                        const htmlCodeBlockMatch = text.match(/```html\s*(<!DOCTYPE html>[\s\S]*?<\/html>)\s*```/);
                        if (htmlCodeBlockMatch && htmlCodeBlockMatch[1]) {
                            return this.processHTMLContent(htmlCodeBlockMatch[1]);
                        }
                        
                        // Then check for regular HTML content
                        const htmlMatch = text.match(/<!DOCTYPE html>[\s\S]*?<\/html>/);
                        if (!htmlMatch) return null;
                        
                        return this.processHTMLContent(htmlMatch[0]);
                    } catch (error) {
                        console.error('Error extracting HTML:', error);
                        return null;
                    }
                },

                processHTMLContent(html) {
                    try {
                        if (!html || !html.includes('<!DOCTYPE html>')) {
                            console.error('Invalid HTML input:', html);
                            return null;
                        }

                        // Extract tailwind config with improved validation
                        let tailwindConfig = '{}';
                        const tailwindConfigMatch = html.match(/tailwind\.config\s*=\s*(\{[\s\S]*?\})/);
                        
                        if (tailwindConfigMatch && tailwindConfigMatch[1]) {
                            // Validate the tailwind config is proper JSON
                            try {
                                const configText = tailwindConfigMatch[1].trim();
                                // Handle unclosed braces
                                const openBraces = (configText.match(/\{/g) || []).length;
                                const closeBraces = (configText.match(/\}/g) || []).length;
                                
                                if (openBraces === closeBraces) {
                                    // Basic validation passed, use the config
                                    tailwindConfig = configText;
                                } else {
                                    console.warn('Tailwind config has mismatched braces, using default');
                                }
                            } catch (e) {
                                console.warn('Error parsing tailwind config, using default:', e);
                            }
                        }
                        
                        // Extract style content
                        let styleContent = '';
                        const styleMatches = html.match(/<style[^>]*>([\s\S]*?)<\/style>/g);
                        if (styleMatches && styleMatches.length > 0) {
                            // Extract content from all style tags and combine them
                            styleMatches.forEach(styleTag => {
                                const contentMatch = styleTag.match(/<style[^>]*>([\s\S]*?)<\/style>/);
                                if (contentMatch && contentMatch[1]) {
                                    styleContent += contentMatch[1] + '\n';
                                }
                            });
                        }

                        // Extract body content
                        let bodyContent = '<body></body>';
                        const bodyMatch = html.match(/<body[^>]*>([\s\S]*?)<\/body>/);
                        if (bodyMatch && bodyMatch[0]) {
                            // Remove any CDN script tags from the body content
                            bodyContent = bodyMatch[0].replace(/https:\/\/cdn[^>]*><\/script>/g, '');
                            
                            // Ensure the body tag is complete
                            if (!bodyContent.includes('</body>')) {
                                bodyContent += '</body>';
                            }
                        }
                        
                        // Using the specified string concatenation method for template creation
                        const newHTML = "<!DOCTYPE html>" +
                            "<html lang=\"en\">" +
                            "<head>" +
                                "<meta charset=\"UTF-8\">" +
                                "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">" +
                                "<title>Generated Component</title>" +
                                "<script src=\"https://cdn.tailwindcss.com\"><\/script>" +
                                "<script>" +
                                    "try { tailwind.config = " + (tailwindConfig || '{}') + " } catch(e) { console.error('Tailwind config error:', e); }" +
                                "<\/script>" +
                                "<style>" +
                                    styleContent +
                                "<\/style>" +
                                "<script defer src=\"https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js\"><\/script>" +
                            "</head>" +
                            (bodyContent || '<body></body>') +
                            "</html>";
                        
                        // Validate the created HTML
                        return this.validateHtml(newHTML) || newHTML;
                    } catch (error) {
                        console.error('Error processing HTML content:', error);
                        return null;
                    }
                },

                validateHtml(html) {
                    try {
                        // Create a DOM parser to parse the HTML string
                        const parser = new DOMParser();
                        const doc = parser.parseFromString(html, 'text/html');
                        
                        // Check for parsing errors
                        const parserErrors = doc.querySelectorAll('parsererror');
                        if (parserErrors.length > 0) {
                            console.error('HTML parsing error:', parserErrors[0].textContent);
                            // Return the original HTML instead of null to ensure we always have content
                            return html;
                        }
                        
                        // Check if essential script tags are present in the parsed document
                        const tailwindScript = doc.querySelector('script[src*="tailwindcss"]');
                        if (!tailwindScript) {
                            console.warn('Tailwind script is missing from the parsed document');
                        }
                        
                        // Ensure the tailwind config section is valid
                        const configScripts = Array.from(doc.querySelectorAll('script'))
                            .filter(script => script.textContent.includes('tailwind.config'));
                        
                        if (configScripts.length === 0) {
                            console.warn('No tailwind config script found in the parsed document');
                        }
                        
                        // If no errors, return the original HTML
                        // This preserves the exact structure without modifying it through formatting
                        return html;
                    } catch (error) {
                        console.error('HTML validation error:', error);
                        // Return the original HTML on error to ensure we always have content
                        return html;
                    }
                },

                // Improved HTML formatting function
                formatHTML(html) {
                    if (!html) return '';
                    
                    try {
                        // Function to recursively format HTML with proper indentation
                        function formatNode(node, level = 0) {
                            const indent = '    '.repeat(level);
                            let result = '';
                            
                            // Handle different node types
                            switch(node.nodeType) {
                                case Node.ELEMENT_NODE:
                                    // Start tag with attributes
                                    result += indent + '<' + node.nodeName.toLowerCase();
                                    
                                    // Add attributes
                                    for (let i = 0; i < node.attributes.length; i++) {
                                        const attr = node.attributes[i];
                                        result += ' ' + attr.name + '="' + attr.value + '"';
                                    }
                                    
                                    // Self-closing tags
                                    if (node.childNodes.length === 0 && ['img', 'br', 'hr', 'input', 'meta', 'link'].includes(node.nodeName.toLowerCase())) {
                                        result += ' />\n';
                                        return result;
                                    }
                                    
                                    result += '>\n';
                                    
                                    // Process children
                                    for (let i = 0; i < node.childNodes.length; i++) {
                                        result += formatNode(node.childNodes[i], level + 1);
                                    }
                                    
                                    // Closing tag
                                    result += indent + '</' + node.nodeName.toLowerCase() + '>\n';
                                    break;
                                    
                                case Node.TEXT_NODE:
                                    // Only add text nodes if they contain non-whitespace content
                                    const text = node.nodeValue.trim();
                                    if (text) {
                                        result += indent + text + '\n';
                                    }
                                    break;
                                    
                                case Node.COMMENT_NODE:
                                    result += indent + '<!-- ' + node.nodeValue + ' -->\n';
                                    break;
                                    
                                case Node.DOCUMENT_TYPE_NODE:
                                    result += '<!DOCTYPE ' + node.name + '>\n';
                                    break;
                            }
                            
                            return result;
                        }
                        
                        // Parse the HTML string
                        const parser = new DOMParser();
                        const doc = parser.parseFromString(html, 'text/html');
                        
                        // Format the document
                        let result = '';
                        
                        // Add DOCTYPE
                        result += '<!DOCTYPE html>\n';
                        
                        // Process the HTML element
                        result += formatNode(doc.documentElement, 0);
                        
                        return result;
                    } catch (error) {
                        console.error('Error formatting HTML:', error);
                        return html; // Return original on error
                    }
                },
                
                escapeHtml(unsafe) {
                    if (!unsafe) return '';
                    return unsafe
                        .replace(/&/g, "&amp;")
                        .replace(/</g, "&lt;")
                        .replace(/>/g, "&gt;")
                        .replace(/"/g, "&quot;")
                        .replace(/'/g, "&#039;");
                },
                
                copyToClipboard(model) {
                    if (!this.modelResponses[model]) return;
                    
                    // Get the response
                    let content = this.modelResponses[model];
                    
                    // Extract HTML if contained within markdown code block
                    const htmlCodeBlockMatch = content.match(/```html\s*(<!DOCTYPE html>[\s\S]*?<\/html>)\s*```/);
                    if (htmlCodeBlockMatch && htmlCodeBlockMatch[1]) {
                        content = htmlCodeBlockMatch[1];
                    } else {
                        // Also check for regular HTML content without code blocks
                        const htmlMatch = content.match(/<!DOCTYPE html>[\s\S]*?<\/html>/);
                        if (htmlMatch) {
                            content = htmlMatch[0];
                        }
                    }
                    
                    navigator.clipboard.writeText(content).then(() => {
                        // Find the button
                        const panel = document.getElementById(`preview-${model}`);
                        if (!panel) return;
                        
                        const copyButton = panel.querySelector('.copy-btn');
                        if (!copyButton) return;
                        
                        // Store original icon
                        const originalIcon = copyButton.innerHTML;
                        
                        // Show success indicator
                        copyButton.innerHTML = '<i class="fas fa-check"></i>';
                        copyButton.classList.add('bg-primary', 'text-white');
                        
                        // Reset after delay
                        setTimeout(() => {
                            copyButton.classList.remove('bg-primary', 'text-white');
                            copyButton.innerHTML = originalIcon;
                        }, 1500);
                        
                        this.showNotification("Copied to clipboard", "success");
                    });
                },
                
                async saveExample(model) {
    if (!this.modelResponses[model]) return;
    
    // Extract HTML if present, otherwise use the full text
    let contentToSave = this.modelResponses[model];
    const htmlMatch = contentToSave.match(/<!DOCTYPE html>[\s\S]*?<\/html>/);
    if (htmlMatch) {
        contentToSave = htmlMatch[0];
    }
    
    // Create confirmation dialog with the Alpine/Tailwind approach
    const dialogDiv = document.createElement('div');
    dialogDiv.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
    
    // Truncate preview if necessary
    const previewContent = contentToSave.length > 200 ? 
        contentToSave.substring(0, 200) + '...' : 
        contentToSave;
    
    dialogDiv.innerHTML = `
        <div class="bg-dark-bg-secondary rounded-lg shadow-lg p-6 max-w-lg w-full mx-4">
            <h3 class="text-lg font-bold text-primary mb-4">Confirm Save</h3>
            <p class="text-dark-text-primary mb-4">Are you sure you want to save this example?</p>
            <div class="bg-dark-bg-tertiary rounded p-3 mb-4 max-h-40 overflow-auto">
                <code class="text-dark-text-secondary text-sm break-all">${this.escapeHtml(previewContent)}</code>
            </div>
            <div class="flex justify-end gap-3">
                <button id="cancel-save" class="px-4 py-2 rounded bg-dark-bg-tertiary hover:bg-dark-border transition-colors">
                    Cancel
                </button>
                <button id="confirm-save" class="px-4 py-2 rounded bg-primary hover:bg-primary-hover text-white transition-colors">
                    Save Example
                </button>
            </div>
        </div>
    `;
    
    document.body.appendChild(dialogDiv);
    
    // Handle button clicks
    document.getElementById('cancel-save').addEventListener('click', () => {
        dialogDiv.remove();
    });
    
    document.getElementById('confirm-save').addEventListener('click', async () => {
        dialogDiv.remove();
        
        try {
            // Read existing examples
            const response = await fetch(this.EXAMPLES_FILE_PATH);
            let examples = [];
            
            if (response.ok) {
                examples = await response.json();
            } else {
                console.warn("Could not load existing examples, creating a new file");
            }
            
            // Add new example
            examples.push({
                "input": this.lastMessage,
                "output": contentToSave
            });
            
            // Save updated examples
            const saveResponse = await fetch(this.EXAMPLES_FILE_PATH, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(examples, null, 4)
            });
            
            if (saveResponse.ok) {
                // Show success visual feedback on the save button
                const panel = document.getElementById(`preview-${model}`);
                if (panel) {
                    const saveButton = panel.querySelector('.save-example-btn');
                    if (saveButton) {
                        const originalIcon = saveButton.innerHTML;
                        saveButton.innerHTML = '<i class="fas fa-check"></i>';
                        saveButton.classList.add('bg-green-600', 'text-white');
                        
                        setTimeout(() => {
                            saveButton.classList.remove('bg-green-600', 'text-white');
                            saveButton.innerHTML = originalIcon;
                        }, 1500);
                    }
                }
                
                this.showNotification("Example saved successfully!", "success");
            } else {
                throw new Error("Failed to save example");
            }
        } catch (error) {
            console.error("Error saving example:", error);
            this.showNotification(`Error saving example: ${error.message}`, "error");
        }
    });
},

showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    
    // Set base classes
    notification.classList.add(
        'fixed', 'top-4', 'right-4', 'px-4', 'py-3', 
        'rounded-lg', 'shadow-lg', 'z-50', 'flex', 
        'items-center', 'gap-2', 'text-white'
    );
    
    // Add type-specific classes
    if (type === 'error') {
        notification.classList.add('bg-red-600');
        notification.innerHTML = '<i class="fas fa-exclamation-circle"></i>';
    } else if (type === 'success') {
        notification.classList.add('bg-green-600');
        notification.innerHTML = '<i class="fas fa-check-circle"></i>';
    } else {
        notification.classList.add('bg-primary');
        notification.innerHTML = '<i class="fas fa-info-circle"></i>';
    }
    
    // Add message
    notification.innerHTML += `<span>${message}</span>`;
    
    // Add to DOM
    document.body.appendChild(notification);
    
    // Add animation
    notification.style.transition = 'opacity 0.5s, transform 0.5s';
    
    // Remove notification after delay
    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateY(-20px)';
        setTimeout(() => notification.remove(), 500);
    }, 3000);
}
}));
});
</script>
</body>
</html>
EOL

    print_success "Web interface created successfully"
}

# ========================================================
# Main Function
# ========================================================

main() {
    # Create log file if it doesn't exist
    touch "${LOG_FILE}" "${ERROR_LOG}"
    
    # Parse command line arguments
    if [[ $# -eq 0 ]]; then
        display_help
        return 1
    fi
    
    case "$1" in
        setup)
            setup
            return $?
            ;;
        finetune)
            # Check for -c flag for custom mode
            if [[ $# -gt 1 && "$2" == "-c" ]]; then
                finetune -c
            else
                finetune
            fi
            return $?
            ;;
        run)
            execute_model
            return $?
            ;;
        help)
            display_help
            return 0
            ;;
        *)
            print_error "Unknown command: $1"
            display_help
            return 1
            ;;
    esac
}

# Display help message
display_help() {
    cat << EOF
LLM Model Fine-tuning and Execution Tool
Usage: ./process.sh [command]

Commands:
    setup            - Set up the environment (install dependencies, create directories)
    finetune         - Fine-tune an LLM model using PEFT
      -c             - Use custom mode with advanced options and performance presets
    run              - Run a fine-tuned model via Ollama
    help             - Display this help message

Examples:
    ./process.sh setup            # Set up the environment
    ./process.sh finetune         # Fine-tune a model with default settings
    ./process.sh finetune -c      # Fine-tune with custom parameter configuration
    ./process.sh run              # Run a fine-tuned model
EOF
}

# Execute main function
main "$@"
exit $?
