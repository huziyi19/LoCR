# A Logical-Enhanced Collaborative Reasoning Framework for Fake News Detection

## Project Overview
This project implements a **fake news detection system** based on large language models (LLMs), integrating logical reasoning and graph neural network (GNN) technologies. The system identifies the authenticity of news content through the following two methods:
1. **Method 1**: LLM-Only Comprehensive Judgment
2. **Method 2**: Full LFND System (LLM + Logical Reasoning + Graph Neural Network + Iterative Correction)

## Core Technologies
- **Natural Language Processing (NLP)**: Utilizes LLMs to evaluate the authenticity of individual sentences.
- **Graph Neural Networks (GNNs)**: Constructs graph structures based on sentence similarity and leverages Graph Convolutional Networks (GCNs) for feature propagation.
- **Logical Reasoning**: Adopts Natural Language Inference (NLI) models to detect logical relationships between sentences.
- **Iterative Correction**: Identifies confidence score fluctuations exceeding thresholds and automatically corrects inconsistent judgments.

## Supported Large Language Models
The system supports the following LLMs via the **OpenRouter API**:
- **deepseek/deepseek-chat** – DeepSeek Chat Model (Recommended Default)
- **openai/gpt-4o-mini** – OpenAI GPT-4o Mini
- **google/gemini-2.5-flash** – Google Gemini 2.5 Flash
- **qwen/qwen3-32b** – Alibaba Cloud Tongyi Qianwen 3 32B Model
- **meta-llama/llama-3.3-70b-instruct** – Meta Llama 3.3 70B Instruction-Tuned Model

You can configure the target model in the `.env` file.

## Project Structure
```
├── data/                   # Data directory
│   ├── gossipcop/          # GossipCop dataset
│   └── politifact/         # PolitiFact dataset
├── LFND.py                 # Main program file
├── load_data.py            # Data loading and preprocessing
├── split_and_detect.py     # Sentence segmentation and confidence detection
├── text_to_graph.py        # Sentence similarity graph construction
├── Logic_propagator.py     # Logical reasoning module
├── GCN_propagator.py       # Graph Convolutional Network module
├── .env.example            # Environment variable template
├── requirements.txt        # Dependencies list
└── README.md               # Project documentation
```

## Installation Instructions
### Experimental Environment
- **Operating System**: Ubuntu 20.04.6 LTS (GNU/Linux 5.15.0-139-generic x86_64)
- **Python Version**: Python 3.11.13
- **GPU**: NVIDIA A100 80GB PCIe (CUDA Version: 13.0, Driver Version: 580.76.05)
- **Network**: Stable internet connection (required for API calls and model downloads)

### Installation Steps
1. Clone the project to your local machine
    ```bash
    git clone <your-project-repository-url>
    cd <your-project-folder>
    ```
2. Create and activate a virtual environment
    ```bash
    # Create virtual environment
    python -m venv venv
    # Activate on Linux/macOS
    source venv/bin/activate
    # Activate on Windows
    venv\Scripts\activate
    ```
3. Install the required dependencies
    ```bash
    pip install -r requirements.txt
    ```
4. Configure environment variables
    ```bash
    cp .env.example .env
    # Edit the .env file and fill in your API key and model selection
    ```

## Usage Instructions
### Data Preparation
1. Place the datasets in the `data/` directory.
2. Supported formats:
   - `.jsonl` format (e.g., GossipCop, PolitiFact)
   - `.csv` format

**Data Format Requirements**:
| Field | Description |
|-------|-------------|
| `text` | Full content of the news article |
| `label` | Ground truth label (`real` or `fake`) |
| `id` | Optional, unique identifier for the news article |
| `title` | Optional, title of the news article |

### Model Configuration
Configure your target LLM in the `.env` file:
```bash
# --- OpenRouter & LLM Configuration ---
OPENROUTER_API_KEY="your-api-key-here"
OPENROUTER_MODEL="qwen/qwen3-32b"
OPENROUTER_API_URL="https://openrouter.ai/api/v1/chat/completions"
```

**List of Optional Models**:
- deepseek/deepseek-chat
- openai/gpt-4o-mini
- google/gemini-2.5-flash
- qwen/qwen3-32b
- meta-llama/llama-3.3-70b-instruct

You can switch between different models by directly modifying the `OPENROUTER_MODEL` field.

### Run Experiments
Execute batch detection with the main program:
```bash
python LFND.py
```
The program will automatically perform the following steps:
1. Load the datasets
2. Run detection methods on each news article
3. Output detailed comparative results
4. Generate a final accuracy report

### Output Results
The program outputs detailed detection results for each sample and a final accuracy comparison across all methods.

## Core Algorithm Description
### 1. Sentence Segmentation and Confidence Evaluation
- Splits news text into semantically complete sentences using LLMs.
- Assigns an authenticity confidence score (ranging from 0 to 1) to each sentence.

### 2. Graph Construction
- Builds graph structures based on the cosine similarity of sentence embeddings.
- Edge weights represent the semantic similarity between connected sentences.

### 3. Logical Reasoning
- Detects entailment and contradiction relationships between sentences using NLI models.
- Implements propagation for two types of logical relationships: **negation** and **entailment**.

### 4. Graph Convolution Propagation
- Performs confidence score propagation on the similarity graph using GCNs.
- Fills in missing confidence values for incomplete nodes.

### 5. Iterative Correction
- Identifies nodes with confidence score changes exceeding predefined thresholds.
- Invokes LLMs to re-evaluate the authenticity of these sentences.
- Iterates until convergence or the maximum number of iterations is reached.

## Dependency Description
**Key Dependencies**:
| Package | Function |
|---------|----------|
| `openai` | OpenAI API client |
| `torch` | PyTorch deep learning framework |
| `torch-geometric` | Graph neural network library |
| `transformers` | Hugging Face Transformers library |
| `sentence-transformers` | Sentence embedding models |
| `networkx` | Graph processing library |
| `numpy` | Numerical computation library |
| `python-dotenv` | Environment variable management tool |

## Notes
1. **API Quota**: Using the OpenRouter API requires a valid API key and sufficient quota.
2. **Model Loading**: Pre-trained models will be downloaded automatically on the first run; a stable network connection is required.
3. **Memory Usage**: Monitor memory consumption when processing large-scale datasets.
4. **GPU Acceleration**: GPU usage is recommended to achieve faster inference speeds.

---

### GitHub Compatibility Optimization Instructions
1. **Hierarchy Standardization**: Adopts `#` for main title, `##` for secondary titles, and `###` for tertiary titles, which can be correctly recognized and rendered by GitHub.
2. **Code Block Formatting**: All command lines and configuration codes use triple backtick ``` formatting, with specified language types (e.g., `bash`), ensuring syntax highlighting.
3. **Table Application**: Uses Markdown tables for data format requirements and dependency descriptions, making the content more organized.
4. **Path and File Naming**: Consistent with GitHub's display habits, folder and file names are clearly marked with comments.
5. **No Special Symbols**: Avoids symbols that are not compatible with Markdown syntax, ensuring normal rendering in the GitHub editor.

