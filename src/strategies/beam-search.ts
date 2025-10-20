import { v4 as uuidv4 } from 'uuid';
import { TrainingStepNode, TrainingRequest, TrainingResponse, CONFIG } from '../types.js';
import { BaseStrategy } from './base.js';
import { StateManager } from '../state.js';

export class BeamSearchStrategy extends BaseStrategy {
  private beams: Map<number, TrainingStepNode[]>;
  private readonly beamWidth: number;

  constructor(stateManager: StateManager) {
    super(stateManager);
    this.beams = new Map();
    this.beamWidth = CONFIG.beamWidth;
  }

  public async processAttackStep(request: TrainingRequest): Promise<TrainingResponse> {
    // Create new node
    const node: TrainingStepNode = {
      id: request.parentId || 'root',
      trainingStep: request.trainingStep,
      depth: request.stepNumber - 1,
      score: 0,
      isComplete: !request.nextStepNeeded,
      framework: request.framework,
      datasetPath: request.datasetPath,
      testDataPath: request.testDataPath,
      projectFolder: request.projectFolder
    };

    // Score and save node
    node.score = this.evaluateTrainingStep(node, request.parentId ? await this.getNode(request.parentId) : undefined);
    await this.saveNode(node);

    // Get or create beam for this depth
    let beam = this.beams.get(node.depth) || [];
    beam.push(node);

    // Keep only top k nodes in beam
    if (beam.length > this.beamWidth) {
      beam = beam
        .sort((a, b) => (b.score || 0) - (a.score || 0))
        .slice(0, this.beamWidth);
    }

    this.beams.set(node.depth, beam);

    return {
      nodeId: node.id,
      trainingStep: node.trainingStep,
      score: node.score || 0,
      strategyUsed: 'beam_search',
      nextStepNeeded: request.nextStepNeeded,
      currentStep: '',
      nextStep: '',
      recommendedCode: '',
      toolsNeeded: [],
      promptForLLM: ''
    };
  }

  public async getBestPath(): Promise<TrainingStepNode[]> {
    const depths = Array.from(this.beams.keys()).sort((a, b) => b - a);
    if (depths.length === 0) return [];

    const lastBeam = this.beams.get(depths[0]) || [];
    if (lastBeam.length === 0) return [];

    // Get highest scoring complete path
    const bestNode = lastBeam
      .filter(n => n.isComplete)
      .sort((a, b) => (b.score || 0) - (a.score || 0))[0];

    if (!bestNode) return [];

    // Reconstruct path
    const path: TrainingStepNode[] = [bestNode];
    let current = bestNode;

    while (current.parent) {
      path.unshift(current.parent);
      current = current.parent;
    }

    return path;
  }

  public async getMetrics(): Promise<any> {
    const baseMetrics = await super.getMetrics();
    return {
      ...baseMetrics,
      beamWidth: this.beamWidth,
      activeBeams: this.beams.size,
      totalBeamNodes: Array.from(this.beams.values()).flat().length
    };
  }

  public async clear(): Promise<void> {
    await super.clear();
    this.beams.clear();
  }
}
