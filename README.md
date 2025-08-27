# Autodidact

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)

Autodidact is an **AI research framework** for building autonomous learning agents. It integrates **perception**, **memory**, **language**, **motor control**, and **reinforcement learning** into a cohesive system capable of self-improvement.

---

## 🚀 Features
- **Modular Architecture** – Each component is isolated for maintainability and scalability.
- **Multi-Modal Perception** – Supports visual, DOM-based, and other sensory inputs.
- **Advanced RL Algorithms** – Includes PPO and other reinforcement learning methods.
- **Meta-Learning** – Optimizes learning strategies over time.
- **Deployment Ready** – Kubernetes and Docker configurations included.

---

## 📂 Project Structure
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

## 📖 Documentation
Full documentation is available in the [`docs/`](docs) directory:
- [Overview](docs/overview.md)
- [Modules](docs/modules.md)
- [Testing](docs/testing.md)
- [Deployment](docs/deployment.md)
- [Contributing](docs/contributing.md)

---

## 🛠 Installation
```bash
git clone https://github.com/isathish/Autodidact.git
cd Autodidact
pip install -r requirements.txt
```

---

## ▶️ Usage
Run the backend:
```bash
python ui/backend/main.py
```
Open the frontend:
```
ui/frontend/index.html
```

---

## 🧪 Testing
Run all tests:
```bash
pytest
```
Run with coverage:
```bash
pytest --cov=.
```

---

## 🤝 Contributing
We welcome contributions! Please read the [Contributing Guide](docs/contributing.md) before submitting pull requests.

---

## 📜 License
This project is licensed under the terms of the [MIT License](LICENSE).
