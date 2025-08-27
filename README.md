# Project Autodidact

**Autonomous Self-Improving AI Agent Framework**

---

## **Overview**
Project Autodidact is a multi-phase research and engineering initiative to build an autonomous AI agent capable of perceiving, acting, reasoning, and improving itself over time. The system integrates perception, motor control, reinforcement learning, semantic understanding, world modeling, grounding, and meta-learning into a unified architecture.

The project is implemented in Python, with modular components for each subsystem, containerized deployment, and a Kubernetes-ready infrastructure.

---

## **Key Features**
- **Multi-Modal Perception**: CNN for visual input, GNN for DOM structure, and cross-attention fusion.
- **Motor Control**: Browser automation and action primitives.
- **Reinforcement Learning**: PPO and Evolution Strategies for adaptive behavior.
- **Semantic Understanding**: Online clustering, token discovery, and predictive modeling.
- **World Modeling**: Recurrent State-Space Models for environment prediction.
- **Grounding**: Cross-modal contrastive learning and knowledge graph population.
- **Meta-Learning**: Goal generation, algorithm tuning, and self-improvement loops.
- **Scalable Deployment**: Dockerized environment with Kubernetes manifests.

---

## **Implementation Roadmap**
The project follows a phased roadmap:

### **Phase 0: Foundation & Setup (Weeks 1–4)**
- Define reward functions
- Set up Python project structure
- Configure Docker sandbox with browser automation
- Integrate ChromaDB and Neo4j
- Curate "Nursery" environment websites

### **Phase 1: Low-Level Perception & Action (Months 2–4)**
- Implement CNN & GNN encoders
- Build motor control API
- Integrate RL loop
- First learning loop with change detection reward

### **Phase 2: Semantics & World Model (Months 5–10)**
- Online clustering for concept formation
- Predictive world model (RSSM/Transformer)
- Token discovery and next-character prediction

### **Phase 3: Grounding & Knowledge Graph (Months 11–18)**
- Cross-modal grounding
- Populate Neo4j with grounded concepts
- Learn relationships from text and actions

### **Phase 4: Meta-Learning & Self-Improvement (Months 19+)**
- Goal generation
- Meta-learning for algorithm tuning
- Self-improvement loop integration

For detailed milestones and modules, see [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md).

---

## **Project Structure**
```
core/           # Core logic and configuration
perception/     # Vision and DOM perception modules
motor/          # Action primitives and control
rl/             # Reinforcement learning agents
concepts/       # Concept formation and clustering
world_model/    # Predictive modeling
language/       # Tokenization and language models
grounding/      # Cross-modal grounding
memory/         # Vector store, graph store, and relationship extraction
goals/          # Goal generation
meta/           # Meta-learning and optimization
training/       # Training loops
ui/             # Frontend and backend interfaces
deploy/         # Deployment scripts and manifests
tests/          # Unit and integration tests
```

---

## **Installation**
### **Prerequisites**
- Python 3.10+
- Docker & Docker Compose
- Kubernetes (optional, for cluster deployment)
- Neo4j & ChromaDB instances

### **Setup**
```bash
# Clone repository
git clone https://github.com/yourusername/project-autodidact.git
cd project-autodidact

# Install dependencies
pip install -r requirements.txt

# (Optional) Build Docker image
docker build -t autodidact:latest -f deploy/k8s/dockerfile .
```

---

## **Usage**
### **Local Development**
```bash
python training/loop_phase1.py
```

### **Docker**
```bash
docker run -it autodidact:latest
```

### **Kubernetes**
```bash
kubectl apply -f deploy/k8s/deployment.yaml
```

---

## **Contributing**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit changes (`git commit -m "Description"`)
4. Push to branch (`git push origin feature-name`)
5. Open a Pull Request

---

## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## **Acknowledgements**
- OpenAI Gym for RL environment inspiration
- PyTorch for deep learning framework
- Neo4j for graph database
- ChromaDB for vector storage
