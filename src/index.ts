import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";
import { Reasoner } from './reasoner.js';
import { ReasoningStrategy, TRAINING_PHASES, FRAMEWORKS } from './types.js';
import * as fs from 'fs';
import * as path from 'path';

export default function({ config }: { config: any }) {
  // Initialize server
  const server = new Server(
    {
      name: "EncoderThinkingMCP",
      version: "1.0.0",
    },
    {
      capabilities: {
        tools: {},
      },
    }
  );

  // Initialize reasoner
  const reasoner = new Reasoner();

  // Helper function to generate step-specific guidance
  function generateStepGuidance(stepNumber: number, framework: string, datasetPath: string, testDataPath: string) {
    const stepGuidance = {
      1: { // Data Preparation
        toolsNeeded: ["pandas", "numpy", "scikit-learn", "matplotlib"],
        recommendedCode: `import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load and normalize data
data = pd.read_csv('${datasetPath}')
# Normalize to 0-1 range
data_normalized = data / 255.0
# Split into train/val/test
train_data, temp_data = train_test_split(data_normalized, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)`,
        promptForLLM: `Create a data loading and preprocessing script for autoencoder training. Load the dataset from '${datasetPath}', normalize pixel values to 0-1 range, and split into train/validation/test sets (80/10/10). Include data visualization to check the preprocessing.`
      },
      2: { // Build Model Architecture
        toolsNeeded: framework === 'pytorch' ? ["torch", "torch.nn"] : ["tensorflow", "keras"],
        recommendedCode: framework === 'pytorch' ? 
          `import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded` :
          `import tensorflow as tf
from tensorflow import keras

def create_autoencoder(input_dim=784, latent_dim=2):
    # Encoder
    encoder = keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(latent_dim, activation='linear')
    ])
    
    # Decoder
    decoder = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(latent_dim,)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(input_dim, activation='sigmoid')
    ])
    
    # Autoencoder
    autoencoder = keras.Model(encoder.input, decoder(encoder.output))
    return autoencoder, encoder, decoder`,
        promptForLLM: `Create an autoencoder model architecture using ${framework}. The model should have an encoder that compresses input to a latent space (2 dimensions for visualization) and a decoder that reconstructs the input. Use appropriate activation functions and layer sizes.`
      },
      3: { // Forward Pass
        toolsNeeded: framework === 'pytorch' ? ["torch"] : ["tensorflow"],
        recommendedCode: framework === 'pytorch' ?
          `# Forward pass
model = Autoencoder()
sample_input = torch.randn(1, 784)  # Example input
with torch.no_grad():
    reconstructed = model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")` :
          `# Forward pass
autoencoder, encoder, decoder = create_autoencoder()
sample_input = tf.random.normal((1, 784))  # Example input
reconstructed = autoencoder(sample_input)
print(f"Input shape: {sample_input.shape}")
print(f"Reconstructed shape: {reconstructed.shape}")`,
        promptForLLM: `Implement a forward pass through the autoencoder model. Test with sample data to ensure the encoder-decoder pipeline works correctly. Print input and output shapes to verify the architecture.`
      },
      4: { // Define Loss Function
        toolsNeeded: framework === 'pytorch' ? ["torch"] : ["tensorflow"],
        recommendedCode: framework === 'pytorch' ?
          `import torch.nn.functional as F

# Define loss function
criterion = nn.MSELoss()

# Example loss calculation
def calculate_loss(model, input_data):
    reconstructed = model(input_data)
    loss = criterion(reconstructed, input_data)
    return loss` :
          `# Define loss function
def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Example loss calculation
def calculate_loss(model, input_data):
    reconstructed = model(input_data)
    loss = mse_loss(input_data, reconstructed)
    return loss`,
        promptForLLM: `Define the loss function for autoencoder training. Use Mean Squared Error (MSE) to compare original input with reconstructed output. Create a function to calculate loss for a batch of data.`
      },
      5: { // Backpropagation
        toolsNeeded: framework === 'pytorch' ? ["torch"] : ["tensorflow"],
        recommendedCode: framework === 'pytorch' ?
          `import torch.optim as optim

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training step
def train_step(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    
    reconstructed = model(data)
    loss = criterion(reconstructed, data)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()` :
          `# Define optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Training step
@tf.function
def train_step(model, data, optimizer):
    with tf.GradientTape() as tape:
        reconstructed = model(data)
        loss = tf.reduce_mean(tf.square(data - reconstructed))
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss.numpy()`,
        promptForLLM: `Implement the backpropagation step for autoencoder training. Set up an optimizer (Adam with learning rate 1e-3) and create a training step function that computes gradients and updates model parameters.`
      },
      6: { // Epochs & Convergence
        toolsNeeded: framework === 'pytorch' ? ["torch"] : ["tensorflow"],
        recommendedCode: framework === 'pytorch' ?
          `# Training loop
def train_autoencoder(model, train_loader, epochs=20):
    model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, data in enumerate(train_loader):
            loss = train_step(model, data, optimizer, criterion)
            epoch_loss += loss
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    
    return losses` :
          `# Training loop
def train_autoencoder(model, train_data, epochs=20):
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_data:
            loss = train_step(model, batch, optimizer)
            epoch_loss += loss
        
        avg_loss = epoch_loss / len(train_data)
        losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    
    return losses`,
        promptForLLM: `Create a training loop that runs for multiple epochs, tracking loss convergence. Include progress logging and return loss history for analysis. Use appropriate batch processing for your data.`
      },
      7: { // Evaluation
        toolsNeeded: ["matplotlib", "seaborn", "sklearn"],
        recommendedCode: `import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Evaluate model
def evaluate_autoencoder(model, test_data):
    model.eval()
    with torch.no_grad():
        # Get latent representations
        latent_representations = model.encoder(test_data)
        
        # Get reconstructions
        reconstructions = model(test_data)
        
        # Calculate reconstruction error
        mse = torch.mean((test_data - reconstructions) ** 2)
        
        return latent_representations, reconstructions, mse.item()

# Visualize latent space
def visualize_latent_space(latent_representations, labels=None):
    plt.figure(figsize=(10, 8))
    if labels is not None:
        scatter = plt.scatter(latent_representations[:, 0], latent_representations[:, 1], 
                            c=labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
    else:
        plt.scatter(latent_representations[:, 0], latent_representations[:, 1], alpha=0.6)
    plt.title('Latent Space Visualization')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.show()`,
        promptForLLM: `Implement model evaluation including reconstruction quality assessment and latent space visualization. Calculate reconstruction error and create visualizations to understand how the model learned to compress data.`
      },
      8: { // Applications
        toolsNeeded: framework === 'pytorch' ? ["torch"] : ["tensorflow"],
        recommendedCode: framework === 'pytorch' ?
          `# Save model
torch.save(model.state_dict(), 'autoencoder_model.pth')
torch.save(model, 'autoencoder_complete.pth')

# Load model
model = Autoencoder()
model.load_state_dict(torch.load('autoencoder_model.pth'))

# Use encoder for feature extraction
def extract_features(model, data):
    model.eval()
    with torch.no_grad():
        features = model.encoder(data)
    return features

# Anomaly detection
def detect_anomalies(model, data, threshold=0.1):
    model.eval()
    with torch.no_grad():
        reconstructed = model(data)
        reconstruction_error = torch.mean((data - reconstructed) ** 2, dim=1)
        anomalies = reconstruction_error > threshold
    return anomalies` :
          `# Save model
model.save('autoencoder_model.h5')

# Load model
model = tf.keras.models.load_model('autoencoder_model.h5')

# Use encoder for feature extraction
def extract_features(encoder, data):
    features = encoder.predict(data)
    return features

# Anomaly detection
def detect_anomalies(model, data, threshold=0.1):
    reconstructed = model.predict(data)
    reconstruction_error = tf.reduce_mean(tf.square(data - reconstructed), axis=1)
    anomalies = reconstruction_error > threshold
    return anomalies`,
        promptForLLM: `Implement model saving/loading and demonstrate applications like feature extraction and anomaly detection. Save the trained model and create functions to use the encoder for dimensionality reduction and the full model for anomaly detection.`
      }
    };

    return stepGuidance[stepNumber] || { toolsNeeded: [], recommendedCode: "", promptForLLM: "" };
  }

  // Helper function to log steps to file
  async function logStepToFile(projectFolder: string, stepNumber: number, phase: string, description: string) {
    if (!projectFolder) return;
    
    const timestamp = new Date().toISOString();
    const logEntry = `Step ${stepNumber}: ${phase} - ${description} [${timestamp}]\n`;
    
    const stepsFile = path.join(projectFolder, 'steps.txt');
    fs.appendFileSync(stepsFile, logEntry);
  }

  // Helper function to log training progress to JSON
  async function logTrainingProgress(projectFolder: string, progressData: any) {
    if (!projectFolder) return;
    
    const logFile = path.join(projectFolder, 'training_log.json');
    
    // Read existing log or create new one
    let logData = [];
    if (fs.existsSync(logFile)) {
      try {
        const existingData = fs.readFileSync(logFile, 'utf8');
        logData = JSON.parse(existingData);
      } catch (error) {
        // If file is corrupted, start fresh
        logData = [];
      }
    }
    
    // Add new progress entry
    logData.push(progressData);
    
    // Write back to file
    fs.writeFileSync(logFile, JSON.stringify(logData, null, 2));
  }

  // Process input and ensure correct types
  function processInput(input: any) {
    const result = {
      trainingStep: String(input.trainingStep || ""),
      stepNumber: Number(input.stepNumber || 0),
      totalSteps: Number(input.totalSteps || 8),
      nextStepNeeded: Boolean(input.nextStepNeeded),
      datasetPath: input.datasetPath || "",
      testDataPath: input.testDataPath || "",
      framework: input.framework || FRAMEWORKS.PYTORCH,
      projectFolder: input.projectFolder || "./autoencoder_project",
      strategyType: input.strategyType as ReasoningStrategy | undefined
    };

    // Validate
    if (!result.trainingStep) {
      throw new Error("trainingStep must be provided");
    }
    if (result.stepNumber < 1 || result.stepNumber > 8) {
      throw new Error("stepNumber must be between 1 and 8");
    }
    if (result.totalSteps !== 8) {
      throw new Error("totalSteps must be 8 for autoencoder training");
    }

    return result;
  }

  // Register the tool
  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: [{
      name: "EncoderThinkingMCP",
      description: "Advanced ML training reasoning tool with multiple strategies including Beam Search and Monte Carlo Tree Search for autoencoder development",
      inputSchema: {
        type: "object",
        properties: {
          trainingStep: {
            type: "string",
            description: "Current training step or action in the autoencoder development"
          },
          stepNumber: {
            type: "integer",
            description: "Current step number in the training pipeline (1-8)",
            minimum: 1,
            maximum: 8
          },
          totalSteps: {
            type: "integer",
            description: "Total expected steps in the training pipeline (always 8)",
            minimum: 8,
            maximum: 8
          },
          nextStepNeeded: {
            type: "boolean",
            description: "Whether another training step is needed"
          },
          datasetPath: {
            type: "string",
            description: "Path to the training dataset"
          },
          testDataPath: {
            type: "string", 
            description: "Path to the test dataset"
          },
          framework: {
            type: "string",
            enum: Object.values(FRAMEWORKS),
            description: "ML framework to use (pytorch, tensorflow, keras)"
          },
          projectFolder: {
            type: "string",
            description: "Folder to save all training artifacts and logs"
          },
          strategyType: {
            type: "string",
            enum: Object.values(ReasoningStrategy),
            description: "Training strategy to use (beam_search or mcts)"
          }
        },
        required: ["trainingStep", "stepNumber", "totalSteps", "nextStepNeeded"]
      }
    }]
  }));

  // Handle requests
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    if (request.params.name !== "EncoderThinkingMCP") {
      return {
        content: [{
          type: "text",
          text: JSON.stringify({ error: "Unknown tool", success: false })
        }],
        isError: true
      };
    }

    try {
      // Process and validate input
      const step = processInput(request.params.arguments);

      // Create project folder if it doesn't exist
      if (step.projectFolder && !fs.existsSync(step.projectFolder)) {
        fs.mkdirSync(step.projectFolder, { recursive: true });
      }

      // Process training step with selected strategy
      const response = await reasoner.processAttackStep({
        attackStep: step.trainingStep,
        attackStepNumber: step.stepNumber,
        totalAttackSteps: step.totalSteps,
        nextAttackStepNeeded: step.nextStepNeeded,
        strategyType: step.strategyType
      });

      // Get training pipeline stats
      const stats = await reasoner.getStats();

      // Generate step-specific guidance
      const currentPhase = TRAINING_PHASES[step.stepNumber - 1];
      const nextPhase = step.stepNumber < 8 ? TRAINING_PHASES[step.stepNumber] : "Training Complete";
      
      // Generate tool recommendations and code prompts
      const { toolsNeeded, recommendedCode, promptForLLM } = generateStepGuidance(
        step.stepNumber, 
        step.framework, 
        step.datasetPath, 
        step.testDataPath
      );

      // Log step to steps.txt
      await logStepToFile(step.projectFolder, step.stepNumber, currentPhase, step.trainingStep);

      // Log training progress to training_log.json
      await logTrainingProgress(step.projectFolder, {
        stepNumber: step.stepNumber,
        phase: currentPhase,
        description: step.trainingStep,
        framework: step.framework,
        datasetPath: step.datasetPath,
        testDataPath: step.testDataPath,
        score: response.score,
        strategyUsed: response.strategyUsed,
        timestamp: new Date().toISOString()
      });

      // Return enhanced response
      const result = {
        stepNumber: step.stepNumber,
        totalSteps: step.totalSteps,
        nextStepNeeded: step.nextStepNeeded,
        trainingStep: step.trainingStep,
        currentStep: currentPhase,
        nextStep: nextPhase,
        nodeId: response.nodeId,
        score: response.score,
        strategyUsed: response.strategyUsed,
        toolsNeeded,
        recommendedCode,
        promptForLLM,
        stats: {
          totalNodes: stats.totalNodes,
          averageScore: stats.averageScore,
          maxDepth: stats.maxDepth,
          branchingFactor: stats.branchingFactor,
          strategyMetrics: stats.strategyMetrics
        }
      };

      return {
        content: [{
          type: "text",
          text: JSON.stringify(result)
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: JSON.stringify({
            error: error instanceof Error ? error.message : String(error),
            success: false
          })
        }],
        isError: true
      };
    }
  });

  return server;
}
