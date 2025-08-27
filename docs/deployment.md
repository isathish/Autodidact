# Deployment

This document explains how to deploy the Autodidact project in different environments.

---

## Prerequisites
- **Docker** installed
- **Kubernetes** cluster (for K8s deployment)
- **kubectl** CLI tool
- **Python 3.8+** installed
- Required Python dependencies from `requirements.txt`

---

## Local Deployment

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Backend
```bash
python ui/backend/main.py
```

### 3. Access the Frontend
Open `ui/frontend/index.html` in your browser.

---

## Docker Deployment

### 1. Build the Docker Image
```bash
docker build -t autodidact:latest -f deploy/k8s/dockerfile .
```

### 2. Run the Container
```bash
docker run -p 8000:8000 autodidact:latest
```

---

## Kubernetes Deployment

### 1. Apply Deployment Configuration
```bash
kubectl apply -f deploy/k8s/deployment.yaml
```

### 2. Check Deployment Status
```bash
kubectl get pods
```

### 3. Expose the Service
If not already exposed in the YAML:
```bash
kubectl expose deployment autodidact --type=LoadBalancer --port=8000
```

---

## Environment Variables
You can configure the deployment using environment variables:
- `PORT`: Port for the backend server
- `DEBUG`: Enable debug mode (`true`/`false`)

---

## Scaling
To scale the deployment:
```bash
kubectl scale deployment autodidact --replicas=3
```

---

## Monitoring
Integrate with monitoring tools like **Prometheus** and **Grafana** for performance tracking.
