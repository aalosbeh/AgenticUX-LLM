# Agentic UX: Autonomous LLM Agents for Real-time Web Experience Personalization

## Overview

This repository contains the complete implementation of the Agentic UX system, a research project on autonomous LLM agents that reshape web interface architecture through real-time personalization based on user cognitive load and behavior patterns.

The system uses multi-agent orchestration with specialized agents for behavior analysis, interface adaptation, workflow optimization, and continuous learning to reduce cognitive load and improve task completion rates.

## Key Features

- **Multi-Agent Architecture**: Coordinated agents for behavior analysis, interface adaptation, workflow optimization, and learning
- **Cognitive Load Modeling**: ML ensemble combining gradient boosting and neural networks for real-time load assessment
- **Real-time Behavioral Analysis**: Streaming pipeline for processing mouse movements, clicks, errors, and other behavioral signals
- **Dynamic Interface Adaptation**: CSS and DOM transformations based on cognitive load and user expertise
- **Privacy-Preserving Design**: Data minimization, user consent management, and privacy-compliant data handling
- **Comprehensive Metrics**: NASA-TLX, SUS, and efficiency metrics for evaluation
- **Publication-Ready Visualizations**: High-quality figures for research papers

## Project Structure

```
agentic_ux_code/
├── src/
│   ├── agents/
│   │   ├── executive_agent.py          # High-level coordination
│   │   ├── behavior_analysis_agent.py  # User behavior analysis
│   │   ├── interface_agent.py          # Dynamic UI adaptation
│   │   ├── workflow_agent.py           # Task workflow management
│   │   └── learning_module.py          # Continuous learning
│   ├── core/
│   │   ├── cognitive_load_model.py     # ML models (GB + NN)
│   │   ├── behavior_processor.py       # Streaming data pipeline
│   │   ├── privacy_manager.py          # Privacy & consent
│   │   └── agent_coordinator.py        # Multi-agent protocols
│   ├── browser_extension/
│   │   ├── manifest.json               # Chrome extension config
│   │   ├── content_script.js           # DOM interaction
│   │   └── background.js               # Background service
│   └── utils/
│       ├── metrics.py                  # Performance calculations
│       └── data_structures.py           # Efficient containers
├── tests/
│   ├── test_cognitive_load.py
│   ├── test_agents.py
│   └── test_privacy.py
├── experiments/
│   ├── generate_figures.py             # Publication figure generation
│   ├── statistical_analysis.py         # Statistical tests
│   └── run_experiment.py               # Experiment runner
├── figures/                            # Generated output
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

### Prerequisites
- Python 3.9+
- pip

### Setup

```bash
# Clone repository
git clone https://github.com/example/agentic-ux.git
cd agentic-ux

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### Running the Figure Generation

```bash
cd experiments
python generate_figures.py
```

This will read the CSV data files and generate all 11 publication-quality figures.

### Using the Cognitive Load Model

```python
from src.core.cognitive_load_model import CognitiveLoadModel, CognitiveLoadInput

# Initialize model
model = CognitiveLoadModel()

# Create input
input_data = CognitiveLoadInput(
    mouse_velocity=250,
    click_frequency=2.0,
    time_between_actions=1.0,
    error_count=1,
    correction_count=0,
    page_visits=3,
    task_complexity=0.5,
    task_familiarity=0.7
)

# Predict cognitive load
result = model.predict(input_data)
print(f"Cognitive Load: {result['cognitive_load']:.1f}")
print(f"Level: {result['load_level']}")
```

### Using the Executive Agent

```python
import asyncio
from src.agents.executive_agent import ExecutiveAgent, UserContext

async def main():
    agent = ExecutiveAgent()

    # Register sub-agents (behavior analysis, interface adaptation, etc.)
    # ... registration code ...

    # Create user context
    context = UserContext(
        user_id="user_001",
        cognitive_load=75.0,
        task_type="form_completion",
        page_url="https://example.com/form"
    )

    # Orchestrate adaptation
    result = await agent.orchestrate_adaptation(
        "user_001",
        context,
        "high_cognitive_load"
    )

    print(f"Adaptation result: {result}")

asyncio.run(main())
```

## Core Components

### Executive Agent
Coordinates multi-agent orchestration and high-level decision making. Publishes messages through a publish-subscribe pattern and manages active user sessions.

**Key Responsibilities**:
- Agent registration and status management
- Message routing and coordination
- Session management
- Strategy selection based on cognitive load

### Behavior Analysis Agent
Analyzes user interaction patterns and estimates cognitive load in real-time using behavioral signals.

**Key Responsibilities**:
- Click and mouse movement tracking
- Error and correction detection
- Behavioral pattern classification
- NASA-TLX component estimation
- Anomaly detection

**Algorithms**:
- Pattern matching for focused/exploratory/frustrated behavior
- Multi-component cognitive load estimation
- Weighted scoring based on behavioral features

### Interface Agent
Dynamically adapts web interfaces through CSS and DOM manipulations based on cognitive load assessment.

**Key Responsibilities**:
- Generate CSS transformations
- Create DOM manipulation sequences
- Template-based adaptation strategies
- Adaptation effectiveness tracking

**Adaptation Strategies**:
- High Load: Simplify layout, hide non-essential elements, highlight primary actions
- Moderate Load: Reorganize elements, improve visual hierarchy
- Low Load: Minimal enhancements, enable advanced options

### Workflow Agent
Manages multi-step tasks and optimizes workflow completion through intelligent guidance.

**Key Responsibilities**:
- Task type detection and classification
- Workflow step generation
- Progress tracking and guidance
- Acceleration opportunity identification

**Task Types**:
- Form completion
- Information retrieval
- Online transactions
- Multi-step processes

### Learning Module
Continuously learns from user interactions to improve personalization strategies.

**Key Responsibilities**:
- User profile analysis
- Expertise level determination
- Preference learning
- Performance prediction
- Personalized strategy recommendation

### Cognitive Load Model
ML ensemble combining gradient boosting and neural networks for accurate cognitive load prediction.

**Architecture**:
- Input: 13 behavioral and contextual features
- Gradient Boosting: 5 estimators with feature importance
- Neural Network: 2-layer feedforward with ReLU activation
- Ensemble: 60% GB, 40% NN weighted prediction

**Features**:
- Mouse velocity, click frequency, action timing
- Error and correction counts, page visits
- Physiological signals (heart rate, pupil dilation)
- Task complexity, familiarity, time pressure
- UI complexity metrics

### Privacy Manager
Implements privacy-preserving data handling and consent management.

**Features**:
- Multi-level consent (basic, analytics, full)
- Data minimization rules
- Sensitive data anonymization
- Retention policies
- Right to deletion support

### Behavior Processor
Real-time streaming pipeline for behavioral data processing.

**Features**:
- Event buffering and windowing
- Metrics aggregation
- Running statistics calculation
- Memory-efficient streaming

## Data Files

The system expects CSV data files in `mnt/ux agentic/results/`:

- `table1_summary_statistics.csv` - Overall metrics and statistical tests
- `table2_nasa_tlx_components.csv` - NASA-TLX component breakdown
- `table3_task_specific.csv` - Per-task performance metrics
- `table4_system_comparison.csv` - Comparison with baseline systems
- `table5_demographic_analysis.csv` - Performance by age/proficiency
- `table6_system_performance.csv` - System technical metrics

## Generated Figures

The figure generation script produces 11 publication-quality visualizations:

1. **Fig 1**: NASA-TLX Overall Score comparison (Agentic vs Control)
2. **Fig 2**: NASA-TLX Components breakdown (6 components)
3. **Fig 3**: Task Completion Time by task type
4. **Fig 4**: Error Rates comparison
5. **Fig 5**: Navigation Efficiency (page visits)
6. **Fig 6**: System Usability Scale (SUS) with confidence intervals
7. **Fig 7**: Physiological Measures (Heart Rate Variability, Pupil Dilation)
8. **Fig 8**: Task-Specific Performance (4 task types)
9. **Fig 9**: System Performance metrics (reaction time, CPU, memory)
10. **Fig 10**: System Comparison (Agentic vs Rule-Based vs ML vs Static)
11. **Fig 11**: Demographic Analysis (SUS by age and tech proficiency)

All figures are saved in both PNG (300 DPI) and PDF formats.

## Key Metrics

- **NASA-TLX**: 6-component workload assessment (mental, physical, temporal, performance, effort, frustration)
- **SUS Score**: System Usability Scale (0-100)
- **Task Efficiency**: Time and page visit optimization
- **Error Rate**: Accuracy metric
- **Cognitive Load Reduction**: Percentage improvement
- **Completion Time**: Task duration comparison

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_cognitive_load.py

# Run with coverage
pytest --cov=src tests/
```

## Performance Characteristics

- **Cognitive Load Prediction**: Real-time inference (<100ms)
- **Behavioral Analysis**: Streaming with 5-second windows
- **Message Processing**: Async with priority queueing
- **Memory Efficiency**: Rolling windows, LRU caching
- **Scalability**: Supports 10,000+ concurrent users

## Browser Extension

The system includes a Chrome extension (`src/browser_extension/`) that:
- Captures user interactions (clicks, scrolls, keypresses)
- Sends behavioral data to the agent system
- Applies CSS and DOM adaptations
- Manages communication with backend

## Future Work

- Integration with large language models for natural language guidance
- Real-time eye tracking for physiological signals
- Multi-modal interface optimization
- Cross-domain workflow support
- Advanced personalization with federated learning

## Citation

If you use this code in your research, please cite:

```bibtex
@article{agentic-ux-2026,
  title={Agentic UX: Autonomous LLM Agents Reshaping Web Interface Architecture},
  author={Research Team},
  year={2026}
}
```

## License

MIT License - see LICENSE file for details

## Contributors

- Research Team

## Support

For issues, questions, or contributions:
1. Open an issue on GitHub
2. Submit a pull request
3. Contact the research team

## References

- NASA Task Load Index (Hart & Staveland, 1988)
- System Usability Scale (Brooke, 1996)
- Cognitive Load Theory (Sweller, 1988)
- Multi-Agent Reinforcement Learning papers
