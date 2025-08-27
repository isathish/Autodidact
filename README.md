# Autodidact Documentation

This documentation provides a detailed overview of the Autodidact project, its modules, and their functionalities.

---

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Modules](#modules)
    - [Concepts](#concepts)
    - [Core](#core)
    - [Deploy](#deploy)
    - [Goals](#goals)
    - [Grounding](#grounding)
    - [Language](#language)
    - [Memory](#memory)
    - [Meta](#meta)
    - [Motor](#motor)
    - [Perception](#perception)
    - [RL](#rl)
    - [Training](#training)
    - [UI](#ui)
    - [Utils](#utils)
    - [World Model](#world-model)
4. [Testing](#testing)
5. [Contributing](#contributing)
6. [License](#license)

---

## Overview
Autodidact is an AI research framework designed for autonomous learning agents. It integrates perception, memory, language, motor control, and reinforcement learning to create self-improving systems.

---

## Project Structure
The project is organized into modular directories, each responsible for a specific aspect of the system.

```
concepts/       # High-level AI concepts and algorithms
core/           # Core reward mechanisms and utilities
deploy/         # Deployment configurations (Kubernetes, Docker)
goals/          # Goal generation and management
grounding/      # Grounding of concepts in sensory data
language/       # Language models and tokenizers
memory/         # Memory graph building and relationship extraction
meta/           # Meta-learning and optimization
motor/          # Motor control and action execution
perception/     # Perception models (vision, DOM graph)
rl/             # Reinforcement learning agents
tests/          # Unit and integration tests
training/       # Training loops and self-improvement
ui/             # User interface (frontend & backend)
utils/          # Utility functions
world_model/    # World modeling and state prediction
```

---

## Modules

### Concepts
Located in `concepts/`, this module contains algorithms for clustering and other conceptual reasoning tasks.

### Core
Located in `core/`, this module defines reward functions and core logic for agent evaluation.

### Deploy
Located in `deploy/`, this module contains deployment configurations for Kubernetes and Docker.

### Goals
Located in `goals/`, this module handles goal generation and prioritization.

### Grounding
Located in `grounding/`, this module maps abstract concepts to sensory data.

### Language
Located in `language/`, this module includes character-level models and tokenizers.

### Memory
Located in `memory/`, this module builds memory graphs and extracts relationships.

### Meta
Located in `meta/`, this module implements meta-learning and optimization strategies.

### Motor
Located in `motor/`, this module controls agent actions and motor outputs.

### Perception
Located in `perception/`, this module processes visual and DOM-based sensory input.

### RL
Located in `rl/`, this module implements reinforcement learning agents such as PPO.

### Training
Located in `training/`, this module contains training loops and self-improvement routines.

### UI
Located in `ui/`, this module provides backend and frontend interfaces for interacting with the system.

### Utils
Located in `utils/`, this module contains helper functions and utilities.

### World Model
Located in `world_model/`, this module predicts future states and models the environment.

---

## Testing
Tests are located in the `tests/` directory. Run them using:
```bash
pytest
```

---

## Contributing
Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request.

---

## License
This project is licensed under the terms of the LICENSE file in the root directory.
