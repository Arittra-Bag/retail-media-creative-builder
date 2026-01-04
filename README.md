# Retail Media Creative Builder

A production-grade AI system for generating compliant retail media creatives at scale. Built with LangGraph for orchestration, this system combines deterministic rule engines with LLM-powered agents to produce platform-ready creative assets that meet strict retailer compliance requirements.

## Overview

The Creative Builder automates the end-to-end workflow of generating retail media creatives, from copy validation through layout planning, image operations, compliance checking, and export optimization. The system is designed for high reliability, auditability, and compliance with retailer-specific guidelines.

### Key Capabilities

- Multi-Agent Orchestration: LangGraph-based workflow with smart retry logic and graceful degradation
- Deterministic Compliance: Rule-based validation against Tesco's Appendix A/B guidelines
- Platform-Aware Generation: Native support for Instagram, Facebook formats with size optimization
- Tool-Based Architecture: Modular, testable components for image operations, compliance checks, and export
- Session Management: MongoDB-backed persistence with full audit trails
- Multi-Format Output: Generate multiple platform variants in a single execution

## Architecture

### System Design

The system follows a multi-agent architecture where specialized agents handle distinct phases of creative generation. Each agent is stateless and deterministic where possible, with LLM calls reserved for creative tasks. Compliance and validation use rule-based logic for consistency and explainability.

### Core Components

#### Agents

- Master Agent: Orchestrates workflow, manages state, handles routing decisions
- Copy Validator: Validates marketing copy against retailer guidelines
- Layout Planner: Generates platform-specific layouts with safe-zone awareness
- Image Operations: Plans image transformations
- Compliance Agent: Validates layouts against Tesco rules
- Exporter: Optimizes file sizes, validates platform requirements
- Summarizer: Generates user-facing summaries and next-step suggestions

#### Tools

Deterministic, testable functions organized by domain:

- Image Operations: remove_bg, resize, crop_rotate, contrast_wcag, compose_layers
- Compliance: detect_copy_issues, check_safe_zones, check_font_sizes, check_overlaps
- Exporters: optimize_filesize, validate_platform_format, render_platform_metadata

#### Graph

LangGraph-based orchestration with smart routing, conditional routing, retry budget, and node-specific retry capabilities.

## Technology Stack

- LangGraph: Agent orchestration and workflow management
- Pydantic: Schema validation and type safety
- MongoDB: Session and turn persistence
- Cerebras: Fast inference for copy generation
- Anthropic: Structured outputs for complex reasoning
- Google Vertex AI: Image generation and multimodal tasks

## Getting Started

### Prerequisites

- Python 3.12 or higher
- MongoDB Atlas account
- API keys for LLM providers

### Installation

1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Configure environment variables in .env file

### Quick Start

```python
from src.api_stub.runner import run_turn

result = run_turn(
    session_id=None,
    user_text="Generate a premium creative for Tesco with packshot.",
    attachments=[],
    ui_context={"selected_formats": ["1080x1080", "1080x1920"]},
    title_if_new="Tesco Demo"
)
```

## Key Features

### Smart Retry Logic

The system implements intelligent retry at multiple stages with retry budget to prevent infinite loops and graceful degradation to summarizer on max attempts.

### Deterministic Compliance

Compliance checks use rule-based logic for consistency. All checks return structured issues with severity, codes, and fix hints.

### Transform Audit Trail

Image operations generate explicit transformation plans stored in MongoDB for full auditability.

### Multi-Format Generation

Generate multiple platform variants in one execution with platform-specific metadata for each format.

## Development

### Adding New Agents

1. Create agent file in src/agents/
2. Implement run_agent function
3. Add node wrapper
4. Update graph topology
5. Add routing logic

### Testing

Each component is independently testable. Run tests with: python scratch_test.py

OR, if production apis are live, then test with (at root):
```
uvicorn run_api:app
```