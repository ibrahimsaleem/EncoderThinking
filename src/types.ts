export interface TrainingStepNode {
  id: string;
  trainingStep: string;
  depth: number;
  score?: number;
  isComplete?: boolean;
  children?: TrainingStepNode[];
  parent?: TrainingStepNode;
  framework?: string;
  datasetPath?: string;
  testDataPath?: string;
  projectFolder?: string;
}

export interface TrainingRequest {
  trainingStep: string;
  stepNumber: number;
  totalSteps: number;
  nextStepNeeded: boolean;
  datasetPath?: string;
  testDataPath?: string;
  framework?: string;
  projectFolder?: string;
  parentId?: string;
  strategyType?: string;
}

export interface TrainingResponse {
  nodeId: string;
  trainingStep: string;
  score: number;
  strategyUsed: string;
  nextStepNeeded: boolean;
  currentStep: string;
  nextStep: string;
  recommendedCode?: string;
  toolsNeeded?: string[];
  promptForLLM?: string;
}

export interface ReasoningStats {
  totalNodes: number;
  averageScore: number;
  maxDepth: number;
  branchingFactor: number;
  strategyMetrics: Record<string, any>;
}

export const CONFIG = {
  beamWidth: 5,
  maxDepth: 10,
  mctsIterations: 50,
  temperature: 0.7, // For training step diversity
  cacheSize: 1000,
  defaultStrategy: 'beam_search'
} as const;

export const TRAINING_PHASES = [
  "Data Preparation",
  "Build Model Architecture", 
  "Forward Pass",
  "Define Loss Function",
  "Backpropagation",
  "Epochs & Convergence",
  "Evaluation",
  "Applications"
] as const;

export const FRAMEWORKS = {
  PYTORCH: 'pytorch',
  TENSORFLOW: 'tensorflow',
  KERAS: 'keras'
} as const;

export enum ReasoningStrategy {
  BEAM_SEARCH = 'beam_search',
  MCTS = 'mcts'
}
