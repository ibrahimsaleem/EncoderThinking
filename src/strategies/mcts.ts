import { v4 as uuidv4 } from 'uuid';
import { TrainingStepNode, TrainingRequest, TrainingResponse, CONFIG } from '../types.js';
import { BaseStrategy } from './base.js';
import { StateManager } from '../state.js';

interface MCTSNode extends TrainingStepNode {
  visits: number;
  ucb1Score?: number;
}

export class MCTSStrategy extends BaseStrategy {
  private readonly explorationConstant = Math.sqrt(2);
  private readonly simulationDepth = 3;

  constructor(stateManager: StateManager) {
    super(stateManager);
  }

  public async processAttackStep(request: TrainingRequest): Promise<TrainingResponse> {
    // Create or get root node
    const rootNode: MCTSNode = {
      id: request.parentId || 'root',
      trainingStep: request.trainingStep,
      depth: request.stepNumber - 1,
      visits: 0,
      score: 0,
      isComplete: !request.nextStepNeeded,
      framework: request.framework,
      datasetPath: request.datasetPath,
      testDataPath: request.testDataPath,
      projectFolder: request.projectFolder
    };

    await this.saveNode(rootNode);

    // Run MCTS iterations
    for (let i = 0; i < CONFIG.mctsIterations; i++) {
      const selectedNode = await this.select(rootNode);
      const expandedNode = await this.expand(selectedNode);
      const simulationScore = await this.simulate(expandedNode);
      await this.backpropagate(expandedNode, simulationScore);
    }

    // Get best child of root
    const bestChild = await this.getBestChild(rootNode);
    
    return {
      nodeId: bestChild.id,
      trainingStep: bestChild.trainingStep,
      score: bestChild.score || 0,
      strategyUsed: 'mcts',
      nextStepNeeded: request.nextStepNeeded,
      currentStep: '',
      nextStep: '',
      recommendedCode: '',
      toolsNeeded: [],
      promptForLLM: ''
    };
  }

  private async select(node: MCTSNode): Promise<MCTSNode> {
    let current = node;
    while (Array.isArray(current.children) && current.children.length > 0) {
      current = await this.selectBestUCB1(current);
    }
    return current;
  }

  private async expand(node: MCTSNode): Promise<MCTSNode> {
    // Create a new training step node as expansion
    const newNode: MCTSNode = {
      id: `${node.id}-${Date.now()}`,
      trainingStep: `Simulated training step at depth ${node.depth + 1}`,
      depth: (node.depth || 0) + 1,
      visits: 0,
      score: 0,
      isComplete: false,
      framework: node.framework,
      datasetPath: node.datasetPath,
      testDataPath: node.testDataPath,
      projectFolder: node.projectFolder
    };

    // Score and save
    newNode.score = this.evaluateTrainingStep(newNode, node);
    await this.saveNode(newNode);

    // Update parent-child relationship
    if (!node.children) node.children = [];
    node.children.push(newNode);
    newNode.parent = node;

    return newNode;
  }

  private async simulate(node: MCTSNode): Promise<number> {
    let current = node;
    let totalScore = current.score || 0;
    
    // Random playout
    for (let depth = 0; depth < this.simulationDepth; depth++) {
      const simulatedNode: MCTSNode = {
        id: `sim-${Date.now()}-${depth}`,
        trainingStep: `Random training simulation at depth ${depth + 1}`,
        depth: (current.depth || 0) + 1,
        visits: 1,
        score: 0,
        isComplete: depth === this.simulationDepth - 1,
        framework: current.framework,
        datasetPath: current.datasetPath,
        testDataPath: current.testDataPath,
        projectFolder: current.projectFolder
      };

      simulatedNode.score = this.evaluateTrainingStep(simulatedNode, current);
      totalScore += simulatedNode.score || 0;
      current = simulatedNode;
    }

    return totalScore / (this.simulationDepth + 1);
  }

  private async backpropagate(node: MCTSNode, score: number) {
    let current: MCTSNode | undefined = node;
    
    while (current) {
      current.visits++;
      if (current.score !== undefined) {
        current.score = ((current.score * (current.visits - 1)) + score) / current.visits;
      }
      current = current.parent as MCTSNode;
    }
  }

  private async selectBestUCB1(node: MCTSNode): Promise<MCTSNode> {
    const children = (node.children || []).filter((c): c is MCTSNode => typeof (c as MCTSNode).visits === 'number');
    const totalVisits = node.visits;
    for (const child of children) {
      const exploitation = (child.score || 0);
      const exploration = Math.sqrt(Math.log(totalVisits) / (child.visits || 1));
      child.ucb1Score = exploitation + this.explorationConstant * exploration;
    }
    return children.reduce((a, b) => (a.ucb1Score || 0) > (b.ucb1Score || 0) ? a : b);
  }

  private async getBestChild(node: MCTSNode): Promise<MCTSNode> {
    const children = (node.children || []).filter((c): c is MCTSNode => typeof (c as MCTSNode).visits === 'number');
    return children.reduce((a, b) => (a.visits > b.visits) ? a : b);
  }

  private calculatePathScore(path: TrainingStepNode[]): number {
    if (path.length === 0) return 0;
    return path.reduce((sum, node) => sum + (node.score || 0), 0) / path.length;
  }

  public async getBestPath(): Promise<TrainingStepNode[]> {
    const nodes = Array.from(this.nodes.values());
    if (nodes.length === 0) return [];

    const completePaths = this.findCompletePaths(nodes);
    return completePaths.reduce((bestPath, currentPath) => 
      this.calculatePathScore(currentPath) > this.calculatePathScore(bestPath)
        ? currentPath
        : bestPath
    );
  }

  private findCompletePaths(nodes: TrainingStepNode[]): TrainingStepNode[][] {
    const endNodes = nodes.filter(n => n.isComplete);
    return endNodes.map(end => this.constructPath(end));
  }

  private constructPath(endNode: TrainingStepNode): TrainingStepNode[] {
    const path: TrainingStepNode[] = [];
    let current: TrainingStepNode | undefined = endNode;
    
    while (current) {
      path.unshift(current);
      current = current.parent;
    }
    
    return path;
  }
}
