# Modules

This document provides detailed descriptions of each module in the Autodidact project.

---

## Concepts (`concepts/`)
Contains algorithms for clustering and conceptual reasoning.  
**Example**: `clustering.py` implements unsupervised learning methods to group similar data points.

---

## Core (`core/`)
Defines reward functions and core logic for agent evaluation.  
**Example**: `rewards.py` calculates reinforcement signals based on agent performance.

---

## Deploy (`deploy/`)
Contains deployment configurations for Kubernetes and Docker.  
**Example**: `deployment.yaml` defines Kubernetes deployment specs.

---

## Goals (`goals/`)
Handles goal generation and prioritization for agents.  
**Example**: `generator.py` creates dynamic objectives based on environment state.

---

## Grounding (`grounding/`)
Maps abstract concepts to sensory data.  
**Example**: `contrastive.py` uses contrastive learning to align concepts with perception.

---

## Language (`language/`)
Includes character-level models and tokenizers.  
**Example**: `char_model.py` implements a neural network for character-based text processing.

---

## Memory (`memory/`)
Builds memory graphs and extracts relationships.  
**Example**: `graph_builder.py` constructs a knowledge graph from agent experiences.

---

## Meta (`meta/`)
Implements meta-learning and optimization strategies.  
**Example**: `optimizer.py` tunes learning parameters dynamically.

---

## Motor (`motor/`)
Controls agent actions and motor outputs.  
**Example**: `actions.py` defines low-level movement commands.

---

## Perception (`perception/`)
Processes visual and DOM-based sensory input.  
**Example**: `vision_cnn.py` implements a convolutional neural network for image recognition.

---

## RL (`rl/`)
Implements reinforcement learning agents such as PPO.  
**Example**: `ppo_agent.py` trains agents using Proximal Policy Optimization.

---

## Training (`training/`)
Contains training loops and self-improvement routines.  
**Example**: `loop_phase1.py` runs the initial training phase.

---

## UI (`ui/`)
Provides backend and frontend interfaces for interacting with the system.  
**Example**: `backend/main.py` runs the API server.

---

## Utils (`utils/`)
Contains helper functions and utilities.

---

## World Model (`world_model/`)
Predicts future states and models the environment.  
**Example**: `rssm.py` implements a Recurrent State-Space Model.
