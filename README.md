# EncoderThinkingMCP

An advanced Model Context Protocol (MCP) server designed to help LLMs think like machine learning engineers and guide step-by-step encoder-decoder development.

**EncoderThinkingMCP** is an adaptation of the PentestThinkingMCP architecture, repurposed for machine learning workflows. It provides:
- Automated ML training path planning using Beam Search and Monte Carlo Tree Search (MCTS)
- Step-by-step reasoning for encoder-decoder development and training
- Training step scoring and prioritization
- Tool recommendations for each step (e.g., PyTorch, TensorFlow, scikit-learn)
- Framework-specific code generation and prompts
- Progress tracking and logging for ML projects

---

## What is EncoderThinkingMCP?

**EncoderThinkingMCP** is an advanced Model Context Protocol (MCP) server designed to empower both human and AI ML engineers. It provides:
- Automated ML training path planning using Beam Search and Monte Carlo Tree Search (MCTS)
- Step-by-step reasoning for encoder-decoder development, training, and evaluation
- Training step scoring and prioritization
- Tool recommendations for each step (e.g., PyTorch, TensorFlow, Keras, scikit-learn)
- Framework-specific code generation and LLM prompts
- Progress tracking and logging for ML projects

---

## Why is it special?

- **Brings LLMs to the next level:** Transforms a normal LLM into a structured, methodical ML engineer and advisor
- **Automates complex ML reasoning:** Finds optimal training sequences, not just single steps
- **Works for any ML framework:** Adapts to PyTorch, TensorFlow, Keras, and other frameworks
- **Bridges the gap between AI and ML engineering:** Makes AI a true partner in machine learning development

---

## Features

- Dual search strategies for ML training modeling:
  - Beam search with configurable width (for methodical training step discovery)
  - MCTS for complex decision spaces (for dynamic training scenarios with unknowns)
- ML-specific scoring and evaluation
- Tree-based training path analysis
- Statistical analysis of potential training vectors
- MCP protocol compliance
- Framework-specific code generation (PyTorch, TensorFlow, Keras)
- Progress tracking and logging

---

## How does it work?

1. **Input:**  
   You (or your AI) provide the current training step/state (e.g., "Start encoder-decoder training with MNIST dataset").
2. **Reasoning:**  
   The server uses Beam Search or MCTS to explore possible next steps, scoring and prioritizing them.
3. **Output:**  
   Returns the next best training step, recommended code, tools needed, and LLM prompt for implementation.

---

## Example Workflow: Training an Autoencoder

1. **Data Preparation:**  
   Input: `trainingStep: "Start encoder-decoder training with MNIST dataset"`  
   Output: `Normalize dataset and split into train/val/test` (tools: pandas, scikit-learn)
2. **Model Architecture:**  
   Input: `trainingStep: "Normalize dataset and split into train/val/test"`  
   Output: `Build encoder-decoder architecture` (tools: PyTorch/TensorFlow)
3. **Forward Pass:**  
   Input: `trainingStep: "Build encoder-decoder architecture"`  
   Output: `Test forward pass through the model` (tools: framework-specific)
4. **Loss Function:**  
   Input: `trainingStep: "Test forward pass through the model"`  
   Output: `Define MSE loss function` (tools: framework-specific)
5. **Training Loop:**  
   Input: `trainingStep: "Define MSE loss function"`  
   Output: `Implement training loop with epochs` (tools: framework-specific)
6. **Evaluation:**  
   Input: `trainingStep: "Implement training loop with epochs"`  
   Output: `Evaluate model and visualize latent space` (tools: matplotlib, seaborn)
7. **Applications:**  
   Input: `trainingStep: "Evaluate model and visualize latent space"`  
   Output: `Save model and implement applications` (tools: framework-specific)

---

## Installation

```sh
git clone https://github.com/ibrahimsaleem/EncoderThinkingMCP.git
cd EncoderThinkingMCP
npm install
npm run build
```

---

## Usage

- Add to your MCP client (Cursor, Claude Desktop, etc.) as a server:
  ```json
  {
    "mcpServers": {
      "EncoderThinkingMCP": {
        "command": "node",
        "args": ["path/to/EncoderThinkingMCP/dist/index.js"]
      }
    }
  }
  ```
- Interact with it by sending training steps and receiving next-step recommendations, code suggestions, and training path guidance.

### Example Usage

```json
{
  "trainingStep": "Start encoder-decoder training with MNIST dataset",
  "stepNumber": 1,
  "totalSteps": 8,
  "nextStepNeeded": true,
  "datasetPath": "./data/mnist.csv",
  "testDataPath": "./data/mnist_test.csv",
  "framework": "pytorch",
  "projectFolder": "./autoencoder_project"
}
```

---

## Search Strategies for ML Training

### Beam Search
- Maintains a fixed-width set of the most promising training paths or model development chains.
- Optimal for step-by-step model development and known ML pattern matching.
- Best for: Enumerating training vectors, methodical model chaining, logical training pathfinding.

### Monte Carlo Tree Search (MCTS)
- Simulation-based exploration of the potential training surface.
- Balances exploration of novel training approaches and exploitation of known techniques.
- Best for: Complex ML projects, scenarios with uncertain outcomes, advanced model development.

---

## Algorithm Details

1. **Training Vector Selection**
   - Beam Search: Evaluates and ranks multiple potential training paths or model development chains.
   - MCTS: Uses UCT for node selection (potential training steps) and random rollouts (simulating training progression).
2. **ML Training Scoring Based On:**
   - Likelihood of successful training
   - Potential model performance
   - Framework compatibility and best practices
   - Strength of connection in a training chain (e.g., data prep enables model training)
3. **Process Management**
   - Tree-based state tracking of training progression
   - Statistical analysis of successful/failed simulated training paths
   - Progress monitoring against ML objectives

---

## Use Cases

- Automated model architecture identification and optimization
- Training pathfinding and optimization
- ML scenario simulation and "what-if" analysis
- Model development strategy refinement
- Assisting in manual ML development by suggesting potential approaches
- Decision tree exploration for complex training vectors
- Strategy optimization for achieving specific ML goals (e.g., feature extraction, anomaly detection)

---

## License

MIT

---

## Parameters and MCP Usage

### Parameters

- **trainingStep** (string, required): Current action/step description.
- **stepNumber** (integer 1-8, required): Current step in the pipeline.
- **totalSteps** (integer = 8, required): Must be 8 for the built-in autoencoder flow.
- **nextStepNeeded** (boolean, required): Whether another step should be proposed.
- **datasetPath** (string, optional): Path to training data file/folder.
- **testDataPath** (string, optional): Path to test data.
- **framework** (string, optional): One of `pytorch`, `tensorflow`, `keras`. Default: `pytorch`.
- **projectFolder** (string, optional): Folder to write logs/artifacts. Default: `./autoencoder_project`.
- **strategyType** (string, optional): One of `beam_search`, `mcts`. Default from config: `beam_search`.

### Server runtime and outputs

- Creates `steps.txt` in `projectFolder` and appends each step summary.
- Appends JSON entries to `training_log.json` in `projectFolder` with scores, strategy, paths, and timestamps.
- Returns enhanced response with `currentStep`, `nextStep`, `toolsNeeded`, `recommendedCode`, and `promptForLLM`.

### Run locally (manual)

```sh
npm install
npm run build
node dist/index.js
```

If using an MCP-aware client, point the client to `node` with `dist/index.js` as the entry as shown in the Usage section.

### Switching strategies and frameworks

- To use Beam Search explicitly: set `"strategyType": "beam_search"`.
- To use MCTS: set `"strategyType": "mcts"`.
- To switch frameworks provide `framework: "tensorflow"` or `"keras"`.

### MCP client hints

- MCP tool name: `EncoderThinkingMCP`.
- Ensure Node.js 16+ is available on the client host.
- On Windows, prefer absolute paths for `datasetPath` if the client sandbox differs from the server.

