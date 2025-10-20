import { TrainingStepNode, TrainingRequest, TrainingResponse, TRAINING_PHASES } from '../types.js';
import { StateManager } from '../state.js';

export interface StrategyMetrics {
  name: string;
  nodesExplored: number;
  averageScore: number;
  maxDepth: number;
  active?: boolean;
  [key: string]: number | string | boolean | undefined; // Allow additional strategy-specific metrics including booleans
}

export abstract class BaseStrategy {
  protected stateManager: StateManager;
  protected nodes: Map<string, TrainingStepNode> = new Map();
  protected readonly minScore = 0.5;

  constructor(stateManager: StateManager) {
    this.stateManager = stateManager;
    this.nodes = new Map();
  }

  abstract processAttackStep(request: TrainingRequest): Promise<TrainingResponse>;
  
  protected async getNode(id: string): Promise<TrainingStepNode | undefined> {
    return this.nodes.get(id);
  }

  protected async saveNode(node: TrainingStepNode): Promise<void> {
    this.nodes.set(node.id, node);
  }

  protected evaluateTrainingStep(node: TrainingStepNode, parent?: TrainingStepNode): number {
    let score = this.minScore;

    // Base scoring
    score += this.calculateMLScore(node, parent);
    
    // Depth penalty
    score -= this.calculateDepthPenalty(node);

    // Parent coherence bonus
    if (parent) {
      score += this.calculateCoherence(parent.trainingStep, node.trainingStep);
    }

    // Step sequence bonus
    score += this.calculateSequenceBonus(node);

    return Math.min(Math.max(score, 0), 1);
  }

  private calculateMLScore(node: TrainingStepNode, parent?: TrainingStepNode): number {
    let score = 0;

    // Length score (up to 0.3)
    score += Math.min(node.trainingStep.length / 200, 0.3);

    // ML-specific keywords
    const mlKeywords = /\b(epoch|loss|gradient|optimizer|batch|model|train|test|validation|accuracy|precision|recall|f1|mse|mae|rmse|r2|latent|encoder|decoder|autoencoder|neural|network|layer|activation|dropout|regularization|normalization|standardization|feature|extraction|dimensionality|reduction|clustering|classification|regression|supervised|unsupervised|reinforcement|learning|deep|shallow|convolutional|recurrent|transformer|attention|backpropagation|forward|pass|backward|pass|weight|bias|parameter|hyperparameter|tuning|grid|search|random|search|bayesian|optimization|cross|validation|k-fold|stratified|holdout|bootstrap|ensemble|bagging|boosting|stacking|voting|pipeline|preprocessing|postprocessing|augmentation|synthetic|data|generation|anomaly|detection|outlier|novelty|drift|concept|drift|data|drift|model|drift|adversarial|robustness|interpretability|explainability|fairness|bias|variance|tradeoff|overfitting|underfitting|generalization|capacity|complexity|regularization|early|stopping|dropout|batch|normalization|layer|normalization|weight|decay|l1|l2|elastic|net|ridge|lasso|svm|random|forest|decision|tree|naive|bayes|knn|k-means|hierarchical|clustering|dbscan|pca|ica|lda|t-sne|umap|manifold|learning|dimensionality|reduction|feature|selection|wrapper|filter|embedded|methods|correlation|mutual|information|chi-square|fisher|score|relief|mrmr|lasso|elastic|net|ridge|regression|logistic|regression|linear|regression|polynomial|regression|support|vector|machine|svm|kernel|trick|rbf|polynomial|sigmoid|linear|kernel|soft|margin|hard|margin|c|parameter|gamma|parameter|degree|parameter|coef0|parameter|nu|parameter|epsilon|parameter|tolerance|parameter|max_iter|parameter|random_state|parameter|verbose|parameter|probability|parameter|class_weight|parameter|sample_weight|parameter|shrinking|parameter|cache_size|parameter|decision_function_shape|parameter|break_ties|parameter|random_state|parameter|verbose|parameter|probability|parameter|class_weight|parameter|sample_weight|parameter|shrinking|parameter|cache_size|parameter|decision_function_shape|parameter|break_ties|parameter)\b/i;
    
    if (mlKeywords.test(node.trainingStep)) {
      score += 0.3;
    }

    // Framework-specific content
    const frameworkKeywords = /\b(pytorch|tensorflow|keras|sklearn|scikit-learn|pandas|numpy|matplotlib|seaborn|plotly|bokeh|altair|dash|streamlit|flask|fastapi|django|tornado|aiohttp|uvicorn|gunicorn|celery|redis|mongodb|postgresql|mysql|sqlite|sqlalchemy|alembic|pydantic|marshmallow|click|typer|rich|tqdm|loguru|structlog|pytest|unittest|coverage|black|flake8|mypy|pre-commit|docker|kubernetes|helm|terraform|ansible|jenkins|github|actions|gitlab|ci|cd|devops|mlops|dataops|gitops|infrastructure|as|code|iac|monitoring|logging|metrics|tracing|observability|apm|rum|synthetic|monitoring|chaos|engineering|disaster|recovery|backup|restore|high|availability|scalability|performance|optimization|caching|load|balancing|auto|scaling|horizontal|scaling|vertical|scaling|microservices|api|gateway|service|mesh|circuit|breaker|bulkhead|timeout|retry|exponential|backoff|jitter|rate|limiting|throttling|quota|throttle|circuit|breaker|bulkhead|timeout|retry|exponential|backoff|jitter|rate|limiting|throttling|quota|throttle)\b/i;
    
    if (frameworkKeywords.test(node.trainingStep)) {
      score += 0.2;
    }

    // Technical content
    if (/[+\-*/=<>]/.test(node.trainingStep)) {
      score += 0.2;
    }

    return score;
  }

  private calculateDepthPenalty(node: TrainingStepNode): number {
    return Math.min((node.depth || 0) * 0.1, 0.3);
  }

  private calculateCoherence(parentTrainingStep: string, childTrainingStep: string): number {
    // Simple word overlap metric
    const parentTerms = new Set(parentTrainingStep.toLowerCase().split(/\W+/));
    const childTerms = childTrainingStep.toLowerCase().split(/\W+/);
    const overlap = childTerms.filter(term => parentTerms.has(term)).length;
    return Math.min(overlap * 0.1, 0.3);
  }

  private calculateSequenceBonus(node: TrainingStepNode): number {
    // Bonus for following the correct ML training sequence
    const stepKeywords = {
      1: ['data', 'load', 'preprocess', 'normalize', 'split', 'train', 'test', 'validation'],
      2: ['model', 'architecture', 'encoder', 'decoder', 'layer', 'neural', 'network'],
      3: ['forward', 'pass', 'encode', 'decode', 'reconstruct', 'test', 'sample'],
      4: ['loss', 'function', 'mse', 'mae', 'criterion', 'error', 'metric'],
      5: ['backprop', 'gradient', 'optimizer', 'adam', 'sgd', 'update', 'weight'],
      6: ['epoch', 'train', 'loop', 'convergence', 'batch', 'iteration'],
      7: ['evaluate', 'test', 'metric', 'accuracy', 'visualize', 'plot'],
      8: ['save', 'load', 'application', 'feature', 'extract', 'anomaly', 'detect']
    };

    const currentStep = node.depth || 1;
    const keywords = stepKeywords[currentStep] || [];
    
    let bonus = 0;
    for (const keyword of keywords) {
      if (node.trainingStep.toLowerCase().includes(keyword)) {
        bonus += 0.1;
      }
    }
    
    return Math.min(bonus, 0.3);
  }

  // Required methods for all strategies
  public async getBestPath(): Promise<TrainingStepNode[]> {
    const nodes = await this.stateManager.getAllNodes();
    if (nodes.length === 0) return [];

    // Default implementation - find highest scoring complete path
    const completedNodes = nodes.filter(n => n.isComplete)
      .sort((a, b) => (b.score ?? 0) - (a.score ?? 0));

    if (completedNodes.length === 0) return [];

    return this.stateManager.getPath(completedNodes[0].id);
  }

  public async getMetrics(): Promise<StrategyMetrics> {
    const nodes = await this.stateManager.getAllNodes();
    
    return {
      name: this.constructor.name,
      nodesExplored: nodes.length,
      averageScore: nodes.length > 0 
        ? nodes.reduce((sum, n) => sum + (n.score ?? 0), 0) / nodes.length 
        : 0,
      maxDepth: nodes.length > 0
        ? Math.max(...nodes.map(n => n.depth))
        : 0
    };
  }

  public async clear(): Promise<void> {
    // Optional cleanup method for strategies
    // Default implementation does nothing
  }
}
