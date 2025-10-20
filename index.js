#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import chalk from 'chalk';

class ReasonerServer {
  constructor() {
    this.trainingSteps = [];
    this.branches = {};
  }

  validateInput(input) {
    const data = input;
    if (!data.trainingStep || typeof data.trainingStep !== 'string') {
      throw new Error('Invalid trainingStep: must be a string');
    }
    if (!data.stepNumber || typeof data.stepNumber !== 'number') {
      throw new Error('Invalid stepNumber: must be a number');
    }
    if (!data.totalSteps || typeof data.totalSteps !== 'number') {
      throw new Error('Invalid totalSteps: must be a number');
    }
    if (typeof data.nextStepNeeded !== 'boolean') {
      throw new Error('Invalid nextStepNeeded: must be a boolean');
    }
    return true;
  }

  formatAttackStep(trainingStepData) {
    const { trainingStepNumber, totalSteps, trainingStep, asset, recommendedTool, critical } = trainingStepData;
    const prefix = critical ? chalk.red('🔥 Critical Path') : chalk.blue('🛡️ Attack Step');
    const header = `${prefix} ${trainingStepNumber}/${totalSteps}`;
    const border = '─'.repeat(Math.max(header.length, trainingStep.length) + 4);
    return `\n┌${border}┐\n│ ${header.padEnd(border.length - 2)} │\n├${border}┤\n│ ${trainingStep.padEnd(border.length - 2)} │\n│ Target Asset: ${asset || 'N/A'}${' '.repeat(Math.max(0, border.length - 15 - (asset ? asset.length : 3)))}│\n│ Recommended Tool: ${recommendedTool || 'N/A'}${' '.repeat(Math.max(0, border.length - 22 - (recommendedTool ? recommendedTool.length : 3)))}│\n└${border}┘`;
  }

  processAttackStep(input) {
    try {
      this.validateInput(input);
      if (input.trainingStepNumber > input.totalSteps) {
        input.totalSteps = input.trainingStepNumber;
      }
      this.trainingSteps.push(input);
      const formattedAttackStep = this.formatAttackStep(input);
      console.error(formattedAttackStep);
      return {
        content: [{
          type: "text",
          text: JSON.stringify({
            trainingStepNumber: input.trainingStepNumber,
            totalSteps: input.totalSteps,
            nextStepNeeded: input.nextStepNeeded,
            trainingStepCount: this.trainingSteps.length,
            asset: input.asset,
            recommendedTool: input.recommendedTool,
            critical: input.critical || false
          }, null, 2)
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: JSON.stringify({
            error: error.message,
            status: 'failed'
          }, null, 2)
        }],
        isError: true
      };
    }
  }
}

const REASONER_TOOL = {
  name: "EncoderThinking",
  description: "An encoder-decoder training reasoning engine that helps LLMs think like ML engineers and guide step-by-step encoder-decoder development",
  inputSchema: {
    type: "object",
    properties: {
      trainingStep: {
        type: "string",
        description: "Current training step or action in the encoder-decoder development"
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
        enum: ["pytorch", "tensorflow", "keras"],
        description: "ML framework to use (pytorch, tensorflow, keras)"
      },
      projectFolder: {
        type: "string",
        description: "Folder to save all training artifacts and logs"
      }
    },
    required: ["trainingStep", "stepNumber", "totalSteps", "nextStepNeeded"]
  }
};

// Initialize MCP server
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

const reasonerServer = new ReasonerServer();

// Register tool listing handler
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [REASONER_TOOL],
}));

// Register tool execution handler
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name === "EncoderThinking") {
    return reasonerServer.processAttackStep(request.params.arguments);
  }
  return {
    content: [{
      type: "text",
      text: `Unknown tool: ${request.params.name}`
    }],
    isError: true
  };
});

// Start the server
async function runServer() {
  const port = process.env.PORT || 3000;
  
  // Check if we should use HTTP/SSE transport (for Smithery) or stdio
  if (process.env.NODE_ENV === 'production' || process.env.SMITHERY_DEPLOYMENT) {
    const transport = new SSEServerTransport('/sse', server);
    await server.connect(transport);
    console.error(`Reasoner MCP Server running on HTTP port ${port}`);
  } else {
    const transport = new StdioServerTransport();
    await server.connect(transport);
    console.error("Reasoner MCP Server running on stdio");
  }
}

runServer().catch((error) => {
  console.error("Fatal error running server:", error);
  process.exit(1);
});