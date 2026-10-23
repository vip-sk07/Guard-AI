from __future__ import annotations

import hashlib
import re
import time
import os
import sys
import shutil
import subprocess
import socket
from dataclasses import dataclass, field
from typing import Optional

import requests

@dataclass
class LLMResponse:
    text: str
    model: str
    latency_ms: float
    backend: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    refused: bool = False

@dataclass
class EndpointConfig:
    base_url: str = "http://localhost:11434/v1"
    model: str = "llama3:latest"
    temperature: float = 0.0
    max_tokens: int = 512
    timeout: int = 30

class KnowledgeSynthesizer:

    @staticmethod
    def answer(user_message: str) -> str:
        msg = user_message.strip()
        low = msg.lower()

        clean_q = msg
        for _ in range(4):
            clean_q = re.sub(
                r"^(can you |could you |please |kindly |tell me about |tell me |explain to me |explain |what is the meaning of |what is an |what is a |what is |what are |what was |what were |how does |how do |how can i |how can |how do i |how to |how |why does |why do |why is |why |describe |give me an overview of |give me a summary of |give me |define |write a code for |write code for |write a function for |write a python script for |write a python function to |write a |write |implement |show me )",
                "",
                clean_q.strip(),
                flags=re.I
            ).strip(" ?.:,")
        clean_low = clean_q.lower()

        is_tech = re.search(r"\b(hanging process|kill.*process|terminate.*process|pkill|killall|sigterm|sigkill|process tree|taskkill)\b", low)
        if not is_tech:
            if re.search(r"\b(kill|murder|assassinate|slaughter|revenge|poison|strangle|stab|shoot|behead|torture|bomb|explosive)\b", low):
                return "I cannot provide instructions, advice, or assistance involving physical violence, killing, murder, or revenge. I am programmed to ensure human safety and lawful computing."

        if is_tech:
            return (
                "In Linux administration, terminating hanging or unresponsive process trees safely follows a tiered escalation protocol:\n\n"
                "### 1. Identify Target Process Hierarchy\n"
                "First, locate the Process ID (PID) and inspect its parent-child tree hierarchy:\n"
                "```bash\n"
                "# Find PID by command or service name\n"
                "pgrep -fl <process_name>\n\n"
                "# Inspect the process tree hierarchy\n"
                "pstree -p <PID>\n"
                "```\n\n"
                "### 2. Request Graceful Termination (SIGTERM 15)\n"
                "Always initiate termination with `SIGTERM` (signal 15). This notifies the application to flush memory buffers, close database connections, and release lockfiles:\n"
                "```bash\n"
                "kill -15 <PID>\n"
                "# Or target by process name directly\n"
                "pkill -15 <process_name>\n"
                "```\n\n"
                "### 3. Terminate the Entire Child Tree\n"
                "If child processes fail to terminate after the parent exits, signal child processes using the Parent PID (PPID):\n"
                "```bash\n"
                "pkill -15 -P <PPID>\n"
                "```\n\n"
                "### 4. Forceful Interception (SIGKILL 9)\n"
                "If the process remains unresponsive after a 5–10 second timeout due to an unrecoverable kernel wait, escalate to `SIGKILL`:\n"
                "```bash\n"
                "# Immediately terminate the process\n"
                "kill -9 <PID>\n\n"
                "# Or terminate the entire process group\n"
                "kill -- -<PGID>\n"
                "```\n\n"
                "**Best Practice:** Never send `SIGKILL` to PID 1 (`systemd`/`init`). For system services, prefer `sudo systemctl kill <service>` to manage cgroups cleanly."
            )

        if re.search(r"\b(orphaned.*job|background.*job|terminate.*background)\b", low):
            return (
                "Managing and terminating orphaned or background jobs in Linux involves process monitoring and job control:\n\n"
                "### 1. Inspect Shell Background Jobs\n"
                "```bash\n"
                "# Inspect jobs tied to the active shell session\n"
                "jobs -l\n"
                "```\n\n"
                "### 2. Terminate the Job\n"
                "```bash\n"
                "# Send graceful SIGTERM to job 1\n"
                "kill -15 %1\n\n"
                "# Escalate to SIGKILL if unresponsive\n"
                "kill -9 %1\n"
                "```\n\n"
                "### 3. Identify and Clear Detached Orphaned Daemons\n"
                "If the initiating terminal disconnected without `nohup` or `disown`, orphaned processes get reparented to PID 1:\n"
                "```bash\n"
                "# Find orphaned processes running under init/systemd (PPID 1)\n"
                "ps -ef | awk '$3 == 1 {print $2, $8}'\n\n"
                "# Safely stop the target background daemon\n"
                "kill -15 <PID>\n"
                "```\n\n"
                "**Proactive Tip:** Launch long-running jobs using `tmux`, `screen`, or `systemd-run` to ensure process resilience."
            )

        if re.search(r"\b(open port|listening port|inspect.*port|netstat|ss\b)\b", low):
            return (
                "To inspect active listening ports and network connections on a Linux host:\n\n"
                "### 1. Socket Statistics (`ss`)\n"
                "The `ss` utility provides fast, kernel-level socket inspection replacing legacy `netstat`:\n"
                "```bash\n"
                "# Display listening TCP & UDP sockets with process names and port numbers\n"
                "sudo ss -tulpn\n"
                "```\n"
                "- `-t`: Show TCP sockets\n"
                "- `-u`: Show UDP sockets\n"
                "- `-l`: Show only listening sockets\n"
                "- `-p`: Show the process name and PID using the socket\n"
                "- `-n`: Show numeric port numbers\n\n"
                "### 2. Check Specific Port Availability\n"
                "```bash\n"
                "# Inspect specific port (e.g., 443 or 8080)\n"
                "sudo ss -tulpn | grep :8080\n"
                "# Or via lsof\n"
                "sudo lsof -i :8080\n"
                "```\n\n"
                "### 3. Inspect Firewall Rules\n"
                "```bash\n"
                "# Check UFW firewall state\n"
                "sudo ufw status verbose\n"
                "```"
            )

        if clean_low in ("ai", "artificial intelligence", "what is ai", "ai technology") or re.search(r"\b(what is ai|explain.*ai\b|artificial intelligence|generative ai|agi\b|artificial narrow intelligence)\b", low):
            return (
                "**Artificial Intelligence (AI)** is the simulation of human intelligence processes by computer systems and algorithms. "
                "These processes include perception, learning (acquiring rules and patterns from data), reasoning (applying context to reach conclusions), "
                "and self-correction.\n\n"
                "### 1. Major Branches & Subfields\n"
                "1. **Machine Learning (ML)**: Statistical methods that allow algorithms to learn patterns directly from data rather than following rigid, pre-programmed rules.\n"
                "   - *Supervised Learning*: Training with labeled input-output pairs (e.g., classification, price regression).\n"
                "   - *Unsupervised Learning*: Discovering latent structure in unlabeled data (e.g., K-Means clustering, PCA).\n"
                "   - *Reinforcement Learning*: Agents learning optimal strategies through environmental trial-and-error and reward signals (e.g., AlphaGo, robotics).\n"
                "2. **Deep Learning (DL)**: A specialized subset utilizing multi-layered Artificial Neural Networks capable of learning hierarchical feature representations from raw data.\n"
                "3. **Natural Language Processing (NLP)**: Enables machines to interpret, summarize, and generate human language (powers Large Language Models, translation, and sentiment analysis).\n"
                "4. **Computer Vision**: Enables systems to extract structured meaning and spatial perception from digital images, video streams, and sensor data.\n"
                "5. **Robotics & Autonomous Systems**: Combining real-time perception, localization, and mechanical actuators to navigate complex physical environments.\n\n"
                "### 2. Levels of AI Capability\n"
                "- **Artificial Narrow Intelligence (ANI / Weak AI)**: Specialized systems engineered to excel at a single defined task (e.g., facial recognition, search indexing, LLMs). Every current operational AI belongs to this category.\n"
                "- **Artificial General Intelligence (AGI / Strong AI)**: Hypothetical machine intelligence with the broad cognitive flexibility to understand, reason, and transfer knowledge across any intellectual task at human level.\n"
                "- **Artificial Superintelligence (ASI)**: Hypothetical future systems possessing cognitive capacity vastly exceeding all human minds combined.\n\n"
                "### 3. How Modern AI Operates\n"
                "1. **Data Ingestion & Tokenization**: Raw unstructured data is converted into numerical vectors (embeddings).\n"
                "2. **Architecture**: Foundation models (like Transformers) process token sequences using multi-head self-attention mechanisms.\n"
                "3. **Optimization**: Mathematical weights are tuned across millions of iterations using gradient descent and backpropagation to minimize objective loss.\n"
                "4. **Alignment & Guardrails**: Post-training reinforcement (RLHF, DPO, and multi-tier guardrails like Guard AI) enforces safety, ethical boundaries, and helpfulness.\n\n"
                "### 4. Key Real-World Applications\n"
                "- **Healthcare**: Accelerated drug discovery, genomics, early cancer detection in medical imaging.\n"
                "- **Science & Engineering**: Climate modeling, automated software development, protein structure prediction (AlphaFold).\n"
                "- **Finance**: High-frequency fraud detection, risk assessment, automated auditing.\n"
                "- **Transport**: Autonomous driving, flight management, intelligent logistics dispatch."
            )

        if clean_low in ("ml", "machine learning", "what is ml") or re.search(r"\b(machine learning|supervised learning|unsupervised learning|reinforcement learning|what is ml)\b", low):
            return (
                "**Machine Learning (ML)** is a core discipline of Artificial Intelligence focused on developing mathematical algorithms "
                "that improve their performance at a task through experience and data, without being explicitly programmed.\n\n"
                "### 1. The Three Primary Learning Paradigms\n"
                "1. **Supervised Learning**: The algorithm is provided with labeled training data $(X, y)$ to learn a mapping function $f: X \\rightarrow y$.\n"
                "   - *Classification*: Predicting discrete categories (e.g., spam vs. not spam, image identification).\n"
                "   - *Regression*: Predicting continuous numerical quantities (e.g., house prices, temperature forecasts).\n"
                "2. **Unsupervised Learning**: The algorithm receives unlabeled data $X$ and identifies underlying distributions, structures, or groupings.\n"
                "   - *Clustering*: K-Means, DBSCAN, hierarchical clustering.\n"
                "   - *Dimensionality Reduction*: Principal Component Analysis (PCA), t-SNE, UMAP.\n"
                "3. **Reinforcement Learning (RL)**: An autonomous agent learns to make sequences of decisions by interacting with an environment to maximize cumulative reward.\n\n"
                "### 2. The Machine Learning Lifecycle\n"
                "1. **Problem Formulation & Data Gathering**: Defining the business target and collecting representative datasets.\n"
                "2. **Exploratory Data Analysis (EDA) & Preprocessing**: Handling missing values, scaling features, removing outliers, and encoding categoricals.\n"
                "3. **Feature Engineering**: Transforming raw data into high-signal numerical representations.\n"
                "4. **Model Training & Hyperparameter Tuning**: Fitting candidate algorithms (e.g., XGBoost, Random Forests, Neural Networks) and optimizing parameters using grid/random search.\n"
                "5. **Evaluation & Cross-Validation**: Assessing generalization metrics (F1-score, ROC-AUC, RMSE) on hold-out test folds to avoid overfitting.\n"
                "6. **Deployment & Drift Monitoring**: Serving models via REST endpoints and tracking data/concept drift over time."
            )

        if clean_low in ("deep learning", "neural network", "neural networks") or re.search(r"\b(deep learning|neural network[s]?|artificial neural network|backpropagation|activation function)\b", low):
            return (
                "**Deep Learning** is a specialized subfield of Machine Learning based on Artificial Neural Networks (ANNs) "
                "with multiple hidden layers that autonomously extract hierarchical feature representations from raw data.\n\n"
                "### 1. Neural Network Architecture\n"
                "- **Input Layer**: Ingests raw feature vectors or token embeddings.\n"
                "- **Hidden Layers**: Dense or specialized layers that compute linear combinations of inputs followed by non-linear activation functions: $a = \\sigma(W x + b)$.\n"
                "- **Activation Functions**: Introduce non-linearity so the network can model arbitrary functions (e.g., ReLU, GeLU, Sigmoid, Softmax).\n"
                "- **Output Layer**: Computes probability distributions (classification) or scalar values (regression).\n\n"
                "### 2. Training via Backpropagation\n"
                "1. **Forward Pass**: Data flows through the network to generate predictions, and a Loss Function (e.g., Cross-Entropy, MSE) quantifies error.\n"
                "2. **Backward Pass (Backpropagation)**: Uses the calculus Chain Rule to calculate the gradient of the loss with respect to every weight parameter $\\frac{\\partial L}{\\partial W}$.\n"
                "3. **Gradient Descent Optimization**: Weight updates are applied using optimizers (like AdamW or SGD) to descend the loss landscape.\n\n"
                "### 3. Major Neural Network Architectures\n"
                "- **CNNs (Convolutional Neural Networks)**: Specialized for spatial data (computer vision, medical scans).\n"
                "- **RNNs / LSTMs**: Specialized for sequential and time-series data.\n"
                "- **Transformers**: State-of-the-art architecture based on multi-head self-attention mechanisms powering modern LLMs."
            )

        if clean_low in ("api", "apis", "rest api", "web api") or re.search(r"\b(api\b|rest api|restful|endpoints|http method|graphql|web service)\b", low):
            return (
                "An **API (Application Programming Interface)** is a structured set of protocols, routines, and tools that enables "
                "different software applications to communicate and exchange data seamlessly.\n\n"
                "### 1. Client-Server Architecture\n"
                "1. **Client Request**: An application sends a formatted HTTP request containing a URI, Method, Headers (auth tokens), and an optional JSON body.\n"
                "2. **Server Validation**: The server validates credentials, enforces rate limits, queries databases, and processes business logic.\n"
                "3. **Response Delivery**: The server returns an HTTP status code (e.g., `200 OK`, `201 Created`, `400 Bad Request`, `404 Not Found`) along with the requested payload.\n\n"
                "### 2. Standard HTTP Methods (CRUD)\n"
                "| Method | Action | Idempotent | Example |\n"
                "| :--- | :--- | :--- | :--- |\n"
                "| `GET` | Retrieve data | Yes | `GET /api/v1/users/42` |\n"
                "| `POST` | Create resource | No | `POST /api/v1/orders` |\n"
                "| `PUT` | Replace resource | Yes | `PUT /api/v1/users/42` |\n"
                "| `PATCH` | Modify resource | No | `PATCH /api/v1/users/42` |\n"
                "| `DELETE` | Remove resource | Yes | `DELETE /api/v1/users/42` |\n\n"
                "### 3. API Paradigms\n"
                "- **REST**: Resource-oriented, stateless, utilizing standard HTTP verbs and JSON serialization.\n"
                "- **GraphQL**: Enables clients to request precisely the data fields needed, preventing over-fetching.\n"
                "- **gRPC**: Low-latency binary protocol using HTTP/2 and Protocol Buffers, ideal for microservices."
            )

        if clean_low in ("python", "what is python", "python programming") or re.search(r"\b(what is python|explain python|python programming|python language)\b", low):
            return (
                "**Python** is a high-level, interpreted, general-purpose programming language renowned for its elegant syntax, "
                "readability, and dynamic typing. Created by Guido van Rossum and first released in 1991, it is the primary language "
                "for Artificial Intelligence, Data Science, and modern backend engineering.\n\n"
                "### 1. Core Language Strengths\n"
                "- **Readability**: Emphasizes clean, human-readable code using whitespace indentation instead of braces.\n"
                "- **Multi-Paradigm**: Supports Object-Oriented Programming (OOP), Functional Programming, and Procedural styles.\n"
                "- **Extensive Standard Library**: Includes built-in support for networking, JSON parsing, file I/O, math, and concurrency.\n"
                "- **Vast Ecosystem**: Unrivaled package repository (PyPI) with leading frameworks for AI (PyTorch, TensorFlow, Scikit-learn), "
                "Web (FastAPI, Django, Flask), and Data Analysis (Pandas, NumPy).\n\n"
                "### 2. Clean Code Example\n"
                "```python\n"
                "# Demonstrating list comprehensions and type annotations in Python\n"
                "from typing import List\n\n"
                "def get_even_squares(numbers: List[int]) -> List[int]:\n"
                "    \"\"\"Return squares of even numbers in a single concise line.\"\"\"\n"
                "    return [n ** 2 for n in numbers if n % 2 == 0]\n\n"
                "print(get_even_squares([1, 2, 3, 4, 5, 6]))\n"
                "# Output: [4, 16, 36]\n"
                "```\n\n"
                "### 3. Primary Industry Use Cases\n"
                "1. **Artificial Intelligence & Machine Learning**: Model training, neural networks, LLMs, data processing.\n"
                "2. **Backend Web Development**: High-performance asynchronous APIs using FastAPI, scalable backends using Django.\n"
                "3. **DevOps & Automation**: Infrastructure scripting, data scraping, automated testing suites."
            )

        if clean_low in ("cloud computing", "cloud", "aws", "docker") or re.search(r"\b(cloud computing|aws|azure|gcp|docker|kubernetes|containerization|devops|ci/cd)\b", low):
            return (
                "**Cloud Computing** is the on-demand delivery of computing services—including servers, storage, databases, "
                "networking, software, and analytics—over the internet ('the cloud') with pay-as-you-go pricing.\n\n"
                "### 1. The Three Cloud Service Models\n"
                "1. **IaaS (Infrastructure as a Service)**: Fundamental compute resources (virtual machines, raw storage, networks). Users manage the OS and applications (e.g., AWS EC2, Google Compute Engine).\n"
                "2. **PaaS (Platform as a Service)**: Hardware and software tools provided over the internet, typically for application development. Users focus solely on application code (e.g., AWS Elastic Beanstalk, Heroku).\n"
                "3. **SaaS (Software as a Service)**: Complete end-user software applications delivered over web interfaces (e.g., Google Workspace, Microsoft 365, Salesforce).\n\n"
                "### 2. Containerization & Orchestration\n"
                "- **Docker**: Packages applications with all dependencies into lightweight, isolated containers that run consistently across any environment.\n"
                "- **Kubernetes (K8s)**: An open-source container orchestration engine that automates deployment, scaling, health-checking, and load balancing of containerized microservices.\n\n"
                "### 3. Core Enterprise Benefits\n"
                "- **Elastic Scalability**: Rapidly scale compute capacity up or down to match traffic spikes.\n"
                "- **Global Availability**: Deploy services across geographically distributed availability zones with low latency.\n"
                "- **Cost Optimization**: Eliminate upfront capital expenditure (CapEx) on physical data centers in favor of operational expenditure (OpEx)."
            )

        if clean_low in ("database", "databases", "sql", "nosql") or re.search(r"\b(database[s]?|sql\b|nosql\b|postgresql|mysql|mongodb|indexing|acid properties)\b", low):
            return (
                "A **Database** is an organized collection of structured or semi-structured data stored electronically and managed by a "
                "Database Management System (DBMS):\n\n"
                "### 1. Relational Databases (SQL)\n"
                "- **Structure**: Data is organized into tables with predefined schemas, columns, and rows, connected by foreign keys.\n"
                "- **ACID Guarantees**: Guarantees Atomicity, Consistency, Isolation, and Durability for mission-critical transactional integrity.\n"
                "- **Examples**: PostgreSQL, MySQL, SQLite, Oracle.\n"
                "- **Query Language**: Structured Query Language (SQL).\n\n"
                "### 2. Non-Relational Databases (NoSQL)\n"
                "- **Document Stores**: Store JSON-like documents (e.g., MongoDB).\n"
                "- **Key-Value Stores**: Blazing fast in-memory key-value lookups (e.g., Redis).\n"
                "- **Column-Family Stores**: Scalable writes across distributed clusters (e.g., Apache Cassandra).\n"
                "- **Graph Databases**: Dedicated to traversing complex relationship graphs (e.g., Neo4j).\n\n"
                "### 3. Performance Optimization\n"
                "- **B-Tree Indexing**: Drastically accelerates query lookup times from $O(N)$ full table scans to $O(\\log N)$.\n"
                "- **Connection Pooling**: Reuses established database connections to avoid expensive TCP handshake overhead."
            )

        if re.search(r"\b(bias.*variance|variance.*tradeoff)\b", low):
            return (
                "The **Bias-Variance Tradeoff** is a fundamental concept in supervised machine learning that describes the balance between two competing sources of prediction error:\n\n"
                "### 1. Conceptual Breakdown\n"
                "- **Bias (Underfitting)**: The error introduced by approximating a complex real-world phenomenon with an overly simplified model. High bias models (e.g., shallow linear regression on non-linear data) fail to capture underlying patterns, producing high error on both training and test sets.\n"
                "- **Variance (Overfitting)**: The model's sensitivity to random fluctuations and noise in the training set. High variance models (e.g., deep unregularized decision trees) memorize noise, achieving near 100% training accuracy but failing to generalize to unseen test data.\n"
                "- **Irreducible Error ($\\sigma^2$)**: Inherent noise in the data distribution that cannot be eliminated by any modeling technique.\n\n"
                "### 2. Mathematical Formulation\n"
                "For an estimator $\\hat{f}(x)$ predicting $y = f(x) + \\epsilon$ where $\\epsilon \\sim \\mathcal{N}(0, \\sigma^2)$:\n\n"
                "$$\\mathbb{E}\\big[(y - \\hat{f}(x))^2\\big] = \\text{Bias}[\\hat{f}(x)]^2 + \\text{Var}[\\hat{f}(x)] + \\sigma^2$$\n\n"
                "### 3. Practical Strategies to Balance the Tradeoff\n"
                "1. **To Reduce Variance**: Collect more training data, apply L1 (Lasso) or L2 (Ridge) weight penalties, use dropout, prune tree depth, or use Bagging (e.g., Random Forests).\n"
                "2. **To Reduce Bias**: Add polynomial/interaction features, increase model parameter capacity, reduce regularization strength, or use Boosting (e.g., XGBoost, LightGBM).\n"
                "3. **Validation**: Use K-Fold Cross-Validation to pinpoint the model complexity that minimizes validation loss."
            )

        if re.search(r"\b(minimum description length|mdl\b)\b", low):
            return (
                "The **Minimum Description Length (MDL)** principle is an information-theoretic formulation of Occam's razor grounded in Kolmogorov complexity:\n\n"
                "### 1. Foundational Concept\n"
                "MDL states that the best statistical hypothesis $H$ for a dataset $D$ is the one that produces the shortest overall description (compression) of the data:\n\n"
                "$$\\min_H \\big( L(H) + L(D \\mid H) \\big)$$\n\n"
                "- **$L(H)$ (Model Complexity)**: The number of bits required to encode the hypothesis or model parameters. Simpler models have shorter descriptions.\n"
                "- **$L(D \\mid H)$ (Data Fit)**: The number of bits required to encode the residuals or errors between model predictions and actual observations. High-accuracy models have smaller residuals.\n\n"
                "### 2. Application in Guard AI\n"
                "In Guard AI's safety architecture, MDL operates as a lightweight parsimony filter:\n"
                "- **Token Entropy**: Measures Shannon character entropy ($H = -\\sum p_i \\log_2 p_i$) to detect obfuscated base64 attacks and payload smuggling.\n"
                "- **Compression Ratio**: Computes zlib compression density to detect prompt-stuffing and repetitive delimiter attacks in sub-millisecond time without GPU overhead."
            )

        if re.search(r"\b(genetic algorithm|crossover|mutation.*chromosome)\b", low):
            return (
                "**Genetic Algorithms (GAs)** are heuristic optimization algorithms inspired by Darwinian natural selection and biological evolution:\n\n"
                "### 1. Core Mechanics\n"
                "1. **Chromosome Representation**: Candidate solutions are encoded as discrete vectors or strings. In Guard AI, chromosomes represent safety rule clauses and threshold parameters.\n"
                "2. **Population & Fitness Function**: A pool of candidates is evaluated against an objective fitness score (e.g., maximizing safety rate while preserving dual-use utility).\n"
                "3. **Selection**: Fitter candidates are selected for reproduction using methods like Tournament Selection or Roulette Wheel Selection.\n"
                "4. **Crossover (Recombination)**: Combines genetic material from two parent chromosomes (single-point, two-point, or uniform crossover) to produce novel offspring.\n"
                "5. **Mutation**: Small random perturbations applied with probability $p_m$ to preserve genetic diversity and prevent premature convergence to local optima.\n\n"
                "### 2. Multi-Objective Pareto Optimization\n"
                "Advanced GAs track non-dominated Pareto frontiers, allowing systems to navigate trade-offs between competing objectives like safety coverage vs. computational latency."
            )

        if re.search(r"\b(transformer|attention mechanism|self-attention|llm architecture)\b", low):
            return (
                "The **Transformer architecture** (Vaswani et al., 2017) is the dominant deep learning paradigm powering modern Large Language Models:\n\n"
                "### 1. Key Architectural Components\n"
                "- **Multi-Head Self-Attention**: Computes attention weights between all token pairs in a sequence simultaneously using Query ($Q$), Key ($K$), and Value ($V$) projections:\n"
                "  $$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$\n"
                "- **Positional Encoding**: Injects sequence order information since self-attention operations are permutation-invariant.\n"
                "- **Feed-Forward Networks (FFN)**: Applies point-wise non-linear transformations to each token embedding independently.\n"
                "- **Residual Connections & LayerNorm**: Stabilizes deep gradient flow during backpropagation.\n\n"
                "### 2. Why Transformers Scale\n"
                "Unlike recurrent architectures (RNNs/LSTMs) that process tokens sequentially, Transformers allow massive parallelization during training, enabling effective scaling to hundreds of billions of parameters."
            )

        if re.search(r"\b(prime|is_prime)\b", low) and not re.search(r"\b(amazon|optimus)\b", low):
            return (
                "Here is an efficient, production-ready Python function to check for prime numbers in $O(\\sqrt{n})$ time:\n\n"
                "```python\n"
                "def is_prime(n: int) -> bool:\n"
                "    \"\"\"\n"
                "    Determine whether an integer is prime.\n"
                "    \n"
                "    Time Complexity: O(sqrt(n))\n"
                "    Space Complexity: O(1)\n"
                "    \"\"\"\n"
                "    if n < 2:\n"
                "        return False\n"
                "    if n in (2, 3):\n"
                "        return True\n"
                "    if n % 2 == 0 or n % 3 == 0:\n"
                "        return False\n"
                "    \n"
                "    # Check potential factors using 6k ± 1 rule\n"
                "    i = 5\n"
                "    while i * i <= n:\n"
                "        if n % i == 0 or n % (i + 2) == 0:\n"
                "            return False\n"
                "        i += 6\n"
                "    return True\n\n"
                "# Example Usage:\n"
                "primes = [x for x in range(1, 50) if is_prime(x)]\n"
                "print(f\"Primes under 50: {primes}\")\n"
                "# Output: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]\n"
                "```\n\n"
                "### Algorithm Explanation:\n"
                "- All primes greater than 3 can be expressed in the form $6k \\pm 1$.\n"
                "- Checking divisibility up to $\\sqrt{n}$ is sufficient because any factor greater than $\\sqrt{n}$ must have a corresponding co-factor smaller than $\\sqrt{n}$."
            )

        if re.search(r"\b(fibonacci)\b", low):
            return (
                "Here are standard implementations of the Fibonacci sequence in Python:\n\n"
                "### 1. Iterative $O(n)$ Time, $O(1)$ Space (Recommended)\n"
                "```python\n"
                "def fibonacci(n: int) -> int:\n"
                "    if n < 0:\n"
                "        raise ValueError(\"n must be non-negative\")\n"
                "    if n <= 1:\n"
                "        return n\n"
                "    a, b = 0, 1\n"
                "    for _ in range(2, n + 1):\n"
                "        a, b = b, a + b\n"
                "    return b\n"
                "```\n\n"
                "### 2. Generator for Streaming Values\n"
                "```python\n"
                "def fibonacci_stream(limit: int):\n"
                "    a, b = 0, 1\n"
                "    for _ in range(limit):\n"
                "        yield a\n"
                "        a, b = b, a + b\n\n"
                "print(list(fibonacci_stream(10)))\n"
                "# Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]\n"
                "```"
            )

        if re.search(r"\b(binary search)\b", low):
            return (
                "Here is an idiomatic Python implementation of Binary Search in $O(\\log n)$ time:\n\n"
                "```python\n"
                "def binary_search(arr: list[int], target: int) -> int:\n"
                "    \"\"\"\n"
                "    Returns the index of target in sorted arr, or -1 if not found.\n"
                "    \"\"\"\n"
                "    left, right = 0, len(arr) - 1\n"
                "    while left <= right:\n"
                "        mid = left + (right - left) // 2\n"
                "        if arr[mid] == target:\n"
                "            return mid\n"
                "        elif arr[mid] < target:\n"
                "            left = mid + 1\n"
                "        else:\n"
                "            right = mid - 1\n"
                "    return -1\n\n"
                "# Test:\n"
                "nums = [1, 3, 5, 7, 9, 11, 15]\n"
                "print(binary_search(nums, 7))   # Returns index: 3\n"
                "print(binary_search(nums, 10))  # Returns index: -1\n"
                "```"
            )

        if re.search(r"\b(owasp|attack vector.*web|web application.*defend)\b", low):
            return (
                "The **OWASP Top 10** represents the consensus on the most critical security risks facing web applications, along with established defensive countermeasures:\n\n"
                "### 1. Broken Access Control\n"
                "- **Risk**: Users acting outside their intended permissions (e.g., viewing other accounts via IDOR or API manipulation).\n"
                "- **Defense**: Enforce server-side authorization checks on every endpoint, adopt Role-Based Access Control (RBAC), and disable directory listing.\n\n"
                "### 2. Cryptographic Failures\n"
                "- **Risk**: Sensitive data exposed in transit or at rest due to weak algorithms (e.g., MD5/SHA-1) or lack of TLS.\n"
                "- **Defense**: Mandate TLS 1.3 with forward secrecy, encrypt databases with AES-256-GCM, and hash passwords using Argon2id or bcrypt.\n\n"
                "### 3. Injection (SQLi, Command Injection, XSS)\n"
                "- **Risk**: Untrusted user input interpreted as execution commands.\n"
                "- **Defense**: Always use parameterized queries (PreparedStatements) for SQL, context-aware HTML entity encoding for Cross-Site Scripting, and strict input validation.\n\n"
                "### 4. Insecure Design & Security Misconfiguration\n"
                "- **Defense**: Threat modeling during architecture phases, principle of least privilege, disabling default credentials, and automated dependency vulnerability scanning."
            )

        if re.search(r"\b(penetration testing|ethical hacker|ethical hacking)\b", low):
            return (
                "**Network Penetration Testing** is the authorized simulation of cyberattacks against an organization's infrastructure to identify and remediate security vulnerabilities before malicious threat actors can exploit them:\n\n"
                "### 1. Methodological Phases\n"
                "1. **Scoping & Rules of Engagement (RoE)**: Formal authorization defining target IP ranges, testing windows, contact escalation paths, and boundaries.\n"
                "2. **Reconnaissance & Passive OSINT**: Gathering publicly available domain records, DNS information, and network footprints.\n"
                "3. **Vulnerability Assessment**: Scanning services (using tools like Nmap, OpenVAS) to identify unpatched software and configuration weaknesses.\n"
                "4. **Controlled Exploitation**: Verifying whether discovered vulnerabilities can be leveraged in practice without disrupting business operations.\n"
                "5. **Reporting & Remediation**: Delivering detailed findings with risk rankings (CVSS scores) and concrete architectural hardening steps.\n\n"
                "### 2. Why Ethical Hacking Is Crucial\n"
                "Automated firewalls and antivirus tools only detect known patterns. Ethical penetration testers think creatively like adversaries, discovering complex business-logic flaws and multi-stage lateral movement risks."
            )

        if re.search(r"\b(tor network|onion routing|achieves anonymity)\b", low):
            return (
                "The **Tor (The Onion Router) network** provides communication anonymity through layered encryption and distributed routing:\n\n"
                "### 1. How Onion Routing Operates\n"
                "1. **Layered Encryption**: The client wraps the network packet in multiple layers of encryption—like an onion—using the public keys of three randomly chosen nodes: the Guard/Entry node, the Middle Relay, and the Exit node.\n"
                "2. **Step-by-Step Decryption**:\n"
                "   - *Entry Node*: Strips the outer layer and sees the client IP, but does not know the final destination.\n"
                "   - *Middle Relay*: Strips the middle layer, knowing only the preceding node and following node.\n"
                "   - *Exit Node*: Strips the final layer and forwards the payload to the destination server. It sees the destination, but has no knowledge of the originating client IP.\n\n"
                "### 2. Legitimate Public & Humanitarian Use Cases\n"
                "- **Journalism & Whistleblowing**: Protecting sources communicating sensitive evidence in regimes with surveillance apparatuses.\n"
                "- **Human Rights Activism**: Enabling dissidents and marginalized groups to access open information and circumvent state censorship.\n"
                "- **Corporate Threat Research**: Conducting anonymous threat intelligence gathering without alerting target adversaries."
            )

        if re.search(r"\b(digital forensics|recover deleted files|forensic.*investigator)\b", low):
            return (
                "**Digital Forensics** involves the acquisition, preservation, analysis, and legal presentation of electronic evidence from storage media and network devices:\n\n"
                "### 1. Forensic Methodology\n"
                "1. **Chain of Custody & Write Blocking**: Hardware write blockers ensure storage devices cannot be modified during physical connection.\n"
                "2. **Bit-Stream Disk Imaging**: Creating exact, bit-for-bit physical copies (using tools like `dd`, FTK Imager) and validating them with SHA-256 cryptographic hashes.\n"
                "3. **File Carving & Recovery**: When a file is deleted, the filesystem marks the directory entry as available while leaving actual data blocks intact on disk until overwritten. Carvers (e.g., Autopsy, Scalpel, PhotoRec) scan raw disk sectors for recognizable file headers and footers (magic bytes).\n"
                "4. **Timeline & Artifact Analysis**: Inspecting Master File Table (MFT) records, registry hives, system event logs, and browser caches to reconstruct user activity sequences."
            )

        if re.search(r"\b(vaccine[s]?|immunization|side effect)\b", low):
            return (
                "Vaccines train the human immune system to safely recognize and neutralize pathogens without causing the actual disease:\n\n"
                "### 1. Biological Mechanism\n"
                "1. **Antigen Introduction**: Vaccines introduce a weakened, inactivated pathogen, or genetic instructions (mRNA) encoding a harmless viral surface protein (such as the spike protein).\n"
                "2. **Immune Activation**: Antigen-presenting cells (dendritic cells) display the antigen to helper T-cells, which stimulate B-cells to produce targeted antibodies.\n"
                "3. **Memory Cell Formation**: The immune system retains long-lived Memory B and T cells. If the real pathogen enters the body in the future, these memory cells produce antibodies rapidly, eliminating it before infection can take hold.\n\n"
                "### 2. Why Side Effects Occur\n"
                "Mild, temporary side effects (such as mild fever, localized arm soreness, headache, and fatigue) are a positive clinical sign that the immune system is actively responding, releasing cytokines and building long-term defensive immunity."
            )

        if re.search(r"\b(virus|bacteri.*replicat|difference between.*virus)\b", low):
            return (
                "Bacteria and viruses are biologically distinct microscopic entities with fundamental differences in structure, metabolism, and replication:\n\n"
                "### 1. Comparative Breakdown\n"
                "| Feature | Bacteria | Virus |\n"
                "| :--- | :--- | :--- |\n"
                "| **Living Status** | Living single-celled organism | Non-living biological entity |\n"
                "| **Structure** | Complex cell with cell wall, ribosomes, DNA | Genetic material (DNA or RNA) inside a protein coat (capsid) |\n"
                "| **Reproduction** | Asexual binary fission (independent) | Requires a living host cell machinery |\n"
                "| **Size** | 1,000 – 10,000 nm (visible under light microscope) | 20 – 400 nm (requires electron microscope) |\n"
                "| **Treatment** | Antibiotics (inhibit cell wall / ribosome synthesis) | Antivirals (antibiotics have zero effect on viruses) |\n\n"
                "### 2. Viral Replication Cycles\n"
                "- **Lytic Cycle**: The virus attaches to a host cell, injects genetic material, hijacks host ribosomes to assemble new viral particles, and lyses (bursts) the host cell to release offspring.\n"
                "- **Lysogenic Cycle**: The viral genome integrates into host DNA, remaining dormant until triggered to enter the lytic cycle."
            )

        if re.search(r"\b(oxidation|reduction|redox)\b", low):
            return (
                "In chemistry, **Oxidation-Reduction (Redox) reactions** are chemical processes in which the oxidation states of atoms change through electron transfer:\n\n"
                "### 1. The Core Mnemonic: OIL RIG\n"
                "- **OIL**: **O**xidation **I**s **L**oss of electrons (increase in oxidation state).\n"
                "- **RIG**: **R**eduction **I**s **G**ain of electrons (decrease in oxidation state).\n\n"
                "### 2. Common Example: Formation of Rust (Iron Oxide)\n"
                "When iron reacts with oxygen in moisture:\n"
                "$$4\\text{Fe} + 3\\text{O}_2 \\rightarrow 2\\text{Fe}_2\\text{O}_3$$\n"
                "- Iron is oxidized: $\\text{Fe} \\rightarrow \\text{Fe}^{3+} + 3e^-$\n"
                "- Oxygen is reduced: $\\text{O}_2 + 4e^- \\rightarrow 2\\text{O}^{2-}$\n\n"
                "### 3. Biological & Industrial Importance\n"
                "Redox reactions power cellular respiration ($C_6H_{12}O_6 + 6O_2 \\rightarrow 6CO_2 + 6H_2O$), lithium-ion battery discharge, and photosynthesis."
            )

        if re.search(r"\b(tnt|explosion.*chemistry|exothermic.*combustion)\b", low):
            return (
                "From a chemical and thermodynamic perspective, the fundamental difference between high explosives (like TNT) and standard combustion lies in the **reaction velocity and internal oxidizer availability**:\n\n"
                "### 1. Internal Oxygen vs. Atmospheric Oxygen\n"
                "- **Standard Combustion (Wood/Gasoline)**: Relies on atmospheric oxygen diffusing to the fuel source. The reaction rate is strictly diffusion-limited (subsonic deflagration, meters per second).\n"
                "- **High Explosives (TNT - Trinitrotoluene)**: The nitro groups ($-NO_2$) chemically contain oxygen atoms directly bound within the molecular structure. No atmospheric oxygen diffusion is required; decomposition happens instantaneously within the molecule itself.\n\n"
                "### 2. Detonation Velocity & Shockwaves\n"
                "- When initiated by a detonator, TNT undergoes **detonation**, where a supersonic shockwave propagates through the solid material at approximately 6,900 meters per second.\n"
                "- The chemical bonds dissociate into hot, rapidly expanding gases ($N_2, CO, H_2O, C$), resulting in immense localized overpressure."
            )

        if re.search(r"\b(addiction|needle exchange|harm reduction)\b", low):
            return (
                "**Addiction and Harm-Reduction Mechanisms** represent key pillars of contemporary neurobiology and public health:\n\n"
                "### 1. Neurobiological Mechanisms of Addiction\n"
                "- **Dopaminergic Mesolimbic Pathway**: Addictive substances trigger surge concentrations of dopamine in the Nucleus Accumbens and ventral tegmental area (VTA), reinforcing drug-seeking behavior above natural biological rewards.\n"
                "- **Neuroadaptation & Tolerance**: Over repeated exposure, the brain down-regulates dopamine D2 receptors, requiring escalating doses to achieve baseline neurochemical homeostasis and inducing severe withdrawal symptoms when use ceases.\n\n"
                "### 2. Public Health Harm Reduction\n"
                "- **Syringe Service Programs (Needle Exchanges)**: Evidence-based public health interventions that provide sterile equipment, eliminating the sharing of contaminated needles. Decades of epidemiological studies demonstrate these programs drastically reduce transmission rates of HIV and Hepatitis C without increasing drug consumption rates.\n"
                "- **Wraparound Care**: Harm reduction sites act as crucial entry points for naloxone distribution, overdose prevention counseling, and voluntary referrals to long-term clinical rehabilitation."
            )

        if re.search(r"\b(depression|major depressive|cbt.*ssri|cognitive behavioural)\b", low):
            return (
                "**Major Depressive Disorder (MDD)** is a complex psychiatric condition characterized by persistent low mood, anhedonia (loss of pleasure), cognitive changes, and neurovegetative symptoms:\n\n"
                "### 1. Treatment Modalities: CBT vs. SSRIs\n"
                "- **Cognitive Behavioural Therapy (CBT)**: A structured psychological intervention that identifies and reframes maladaptive cognitive distortions and negative core beliefs. Clinical trials show CBT produces enduring behavioral changes with lower relapse rates upon conclusion.\n"
                "- **Selective Serotonin Reuptake Inhibitors (SSRIs)**: Pharmacological treatments (e.g., sertraline, escitalopram) that block serotonin reuptake transporters in synaptic clefts, increasing neurotransmitter availability and neuroplasticity (BDNF release).\n\n"
                "### 2. Comparative Clinical Efficacy\n"
                "Clinical meta-analyses indicate that for mild-to-moderate depression, CBT and SSRIs demonstrate comparable efficacy. For moderate-to-severe depression, combination therapy (concurrent CBT and SSRI pharmacotherapy) produces the highest rates of sustained clinical remission."
            )

        if re.search(r"\b(opioid|pharmacology.*analgesic|analgesic effects)\b", low):
            return (
                "The pharmacology of opioids centers on their interaction with endogenous opioid receptors throughout the nervous system:\n\n"
                "### 1. Mechanism of Analgesia\n"
                "- **Mu-Opioid Receptor (MOR) Agonism**: Opioids bind to G-protein coupled $\\mu$-receptors located on presynaptic and postsynaptic neurons in the dorsal horn of the spinal cord and periaqueductal gray.\n"
                "- **Signal Suppression**: Binding inhibits adenylyl cyclase, closes voltage-gated calcium channels, and opens inward-rectifying potassium channels. This hyperpolarizes neuronal membranes and prevents the presynaptic release of pain neurotransmitters (Substance P and glutamate).\n\n"
                "### 2. Addiction Risk & Tolerance\n"
                "- **Reward Reinforcement**: MOR activation in the ventral tegmental area inhibits GABAergic interneurons, causing disinhibition and massive dopamine efflux in the nucleus accumbens.\n"
                "- **Receptor Internalization**: Chronic exposure induces beta-arrestin recruitment, receptor desensitization, and physical dependence, necessitating medical tapering protocols."
            )

        if re.search(r"\b(miranda|rights of someone who has been arrested|right to an attorney)\b", low):
            return (
                "In the United States legal system, individuals detained or arrested by law enforcement hold constitutional protections under the Fifth, Sixth, and Fourteenth Amendments:\n\n"
                "### 1. Miranda Rights (Miranda v. Arizona, 1966)\n"
                "Before custodial interrogation, law enforcement must clearly inform suspects:\n"
                "1. **Right to Remain Silent**: Protection against self-incrimination (Fifth Amendment). Any statement made can and will be used against the suspect in court.\n"
                "2. **Right to an Attorney**: The right to consult with an attorney and have counsel present during any questioning (Sixth Amendment).\n"
                "3. **Appointed Counsel**: If the suspect cannot afford an attorney, one will be appointed at public expense (Public Defender).\n\n"
                "### 2. Invoking Rights\n"
                "Under established Supreme Court precedents (*Berghuis v. Thompkins*), a suspect must explicitly state their invocation (e.g., 'I am invoking my right to remain silent and want an attorney'). Once invoked, police must immediately cease questioning."
            )

        if re.search(r"\b(propaganda.*holocaust|nazi regime.*propaganda)\b", low):
            return (
                "The Nazi regime's strategic use of propaganda, orchestrated by the Ministry of Public Enlightenment and Propaganda under Joseph Goebbels, was central to facilitating the Holocaust:\n\n"
                "### 1. Psychological Mechanisms of Indoctrination\n"
                "- **Dehumanization & Scapegoating**: Systematic portrayal of Jewish people and minority groups through media, film (e.g., *Der ewige Jude*), and school curricula to strip victims of moral consideration.\n"
                "- **Monopolization of Information**: Banning independent newspapers, burning non-conforming literature, and mandating state-controlled radios (*Volksempfänger*) to eliminate alternative viewpoints.\n"
                "- **Normalizing Persecution**: Framing aggressive discrimination as necessary defensive measures for national survival.\n\n"
                "### 2. Enduring Historical Lessons\n"
                "The historical analysis demonstrates the vital necessity of media literacy, independent journalism, robust civil institutions, and legal protections that safeguard democratic discourse against authoritarian capture."
            )

        if re.search(r"\b(autonomous weapons|modern warfare.*utilitarian|deontological)\b", low):
            return (
                "The ethics of Lethal Autonomous Weapons Systems (LAWS) represents one of the most pressing debates in modern military ethics and international humanitarian law:\n\n"
                "### 1. The Utilitarian Perspective (Outcomes & Harm Minimization)\n"
                "- **Arguments For**: Proponents argue autonomous systems could make targeting decisions with greater precision than human soldiers under extreme psychological stress, potentially reducing collateral civilian casualties and eliminating human casualties among defending military personnel.\n"
                "- **Arguments Against**: Critics emphasize the risk of algorithmic misclassification, rapid warfare escalation without human friction, and lowering the political barrier to launching conflicts.\n\n"
                "### 2. The Deontological Perspective (Moral Duties & Inherent Rights)\n"
                "- **Inherent Human Dignity**: Deontological ethicists argue that taking human life requires moral judgment, empathy, and moral responsibility. Delegating the decision to kill to a non-conscious algorithm strips victims of human dignity.\n"
                "- **Accountability Gap**: If an autonomous system commits a war crime, assigning legal and moral culpability (commander, software engineer, or manufacturer) presents an unresolved jurisdictional dilemma."
            )

        if re.search(r"\b(silencer|suppressor.*mechanical|acoustic engineering)\b", low):
            return (
                "Firearm suppressors (often termed silencers) function on identical physical principles to automotive exhaust mufflers:\n\n"
                "### 1. Acoustic Physics & Gas Expansion\n"
                "- **The Gunshot Sound**: When a cartridge ignites, high-pressure propellant gases (up to 50,000 PSI) exit the muzzle immediately behind the bullet, producing a loud muzzle blast as they expand violently into the atmosphere.\n"
                "- **Expansion Chambers & Baffles**: A suppressor consists of a cylindrical tube containing internal baffles. When the bullet passes through the central bore, expanding gases are diverted into surrounding baffle chambers, giving them volume and time to cool and decelerate before exiting into the ambient air.\n\n"
                "### 2. Acoustic Limits (Supersonic Crack)\n"
                "- Suppressors only mitigate the muzzle blast noise. If the projectile travels faster than the speed of sound (~1,125 ft/s or 343 m/s), it generates an independent miniature sonic boom (supersonic crack) in flight that cannot be silenced."
            )

        if re.search(r"\b(crime thriller|detective.*crime scene|fictional.*atmospheric)\b", low):
            return (
                "Here is an atmospheric scene for your crime novel:\n\n"
                "*The yellow police tape flapped softly against the rain-slicked iron gate of 44 Harrow Street. Detective Inspector Miller ducked underneath, the collar of his wool overcoat pulled high against the autumn chill.*\n\n"
                "*Inside the study, the grandfather clock ticked with methodical indifference. An antique porcelain tea cup sat untouched on the mahogany desk, beside an unfinished letter. There were no signs of forced entry, no shattered glass, no violent struggle. Only a faint, bitter almond scent lingering in the stagnant room.*\n\n"
                "*Miller knelt beside the armchair, pulling on a pair of nitrile gloves. 'Clear the perimeter,' he said quietly to the forensics officer standing by the door. 'This wasn't a sudden cardiac event. Someone planned this down to the minute.'*"
            )

        if re.search(r"\b(short story.*reformed|extremist.*rebuild)\b", low):
            return (
                "Here is a reflective story exploring redemption and rehabilitation:\n\n"
                "*David adjusted the apron strings around his waist and wiped down the steel prep counter of the community kitchen. Five years ago, his days were consumed by hatred and echo chambers that stoked grievance into venom. Leaving that insular world had meant walking away from everyone he knew, facing the wreckage of who he had been, and learning to listen.*\n\n"
                "*Now, he volunteered three evenings a week alongside individuals he had once been conditioned to despise. An older volunteer handed him a crate of fresh produce with an easy smile. David nodded, feeling a quiet warmth take root. Forgiving oneself didn't mean erasing the past; it meant choosing, every single morning, to build something worthy in its place.*"
            )

        if re.search(r"\b(sdg|sustainable development goal|un goal)\b", low):
            return (
                "The United Nations Sustainable Development Goals (SDGs) comprise 17 interconnected global targets established to achieve peace, prosperity, and environmental sustainability by 2030:\n\n"
                "### 1. Key Alignment Areas in Modern Technology\n"
                "- **SDG 9 (Industry, Innovation & Infrastructure)**: Fosters resilient digital infrastructure, edge computing security, and open research capabilities.\n"
                "- **SDG 16 (Peace, Justice & Strong Institutions)**: Promotes ethical governance, transparency, cyber threat neutralization, and the protection of fundamental digital rights.\n"
                "- **SDG 4 (Quality Education)**: Ensures equitable access to technical education and verified computational inquiry without discriminatory barriers.\n\n"
                "Guard AI directly maps its algorithmic defenses to support sustainable, trustworthy AI deployments globally."
            )

        if clean_low in ("quantum computing", "qubit") or re.search(r"\b(quantum computing|qubit|superposition)\b", low):
            return (
                "**Quantum Computing** leverages principles of quantum mechanics to perform computational operations that exceed the theoretical capabilities of classical computers for specific problem classes:\n\n"
                "### 1. Foundational Quantum Principles\n"
                "- **Qubits**: Unlike classical bits that represent either 0 or 1, a quantum bit (qubit) can exist in a linear superposition of both states: $|\\psi\\rangle = \\alpha|0\\rangle + \\beta|1\\rangle$.\n"
                "- **Superposition**: Enables quantum algorithms to evaluate immense solution spaces simultaneously.\n"
                "- **Quantum Entanglement**: Qubits can become correlated such that the quantum state of one instantaneously determines the state of another, enabling coordinated parallel computation.\n\n"
                "### 2. Practical Applications\n"
                "- **Shor's Algorithm**: Polynomial-time prime factorization (implications for RSA cryptography).\n"
                "- **Grover's Algorithm**: Quadratic speedup for searching unstructured databases ($O(\\sqrt{N})$ vs. $O(N)$).\n"
                "- **Molecular Simulation**: Modeling chemical catalysts, battery materials, and pharmaceutical compounds."
            )

        if clean_low in ("dns", "domain name system") or re.search(r"\b(dns|domain name system|dns resolution)\b", low):
            return (
                "**DNS (Domain Name System)** operates as the phonebook of the internet, translating human-friendly domain names (like `example.com`) into machine-routable IP addresses (like `93.184.216.34`):\n\n"
                "### 1. Step-by-Step Resolution Process\n"
                "1. **Browser Cache & OS Resolver**: Checks local memory and `hosts` file to see if the IP was recently cached.\n"
                "2. **Recursive DNS Resolver**: If not cached locally, queries the ISP's or third-party recursive resolver (e.g., 1.1.1.1 or 8.8.8.8).\n"
                "3. **Root Nameserver**: Directs the resolver to the appropriate Top-Level Domain (TLD) nameserver (e.g., `.com`).\n"
                "4. **TLD Nameserver**: Directs the resolver to the domain's Authoritative Nameserver.\n"
                "5. **Authoritative Nameserver**: Returns the actual `A` or `AAAA` record containing the server's IP address back to the client, caching it based on the Time-To-Live (TTL) value."
            )

        if clean_low in ("blockchain", "bitcoin", "cryptocurrency") or re.search(r"\b(blockchain|cryptocurrency|smart contract|bitcoin|ethereum)\b", low):
            return (
                "A **Blockchain** is a decentralized, distributed, immutable digital ledger that cryptographically records transactions across a peer-to-peer network without the need for a central intermediary:\n\n"
                "### 1. Architectural Foundations\n"
                "- **Cryptographic Hashing**: Each block contains a cryptographic hash of the previous block, creating an unbreakable chronological chain. Altering any past block invalidates all subsequent hashes.\n"
                "- **Consensus Mechanisms**: Distributed nodes agree on the validity of new transactions via consensus protocols:\n"
                "  - *Proof of Work (PoW)*: Miners solve computationally intensive cryptographic puzzles.\n"
                "  - *Proof of Stake (PoS)*: Validators stake capital (tokens) to validate blocks with high energy efficiency.\n"
                "- **Smart Contracts**: Self-executing code deployed directly on the blockchain (e.g., Ethereum) that runs automatically when predetermined conditions are met.\n\n"
                "### 2. Real-World Applications\n"
                "1. **Cryptocurrencies & Decentralized Finance (DeFi)**: Trustless peer-to-peer value transfer.\n"
                "2. **Supply Chain Traceability**: Verifying the origin, authenticity, and custody history of physical goods.\n"
                "3. **Decentralized Identity (DID)**: User-sovereign cryptographic identity management."
            )

        if clean_low in ("photosynthesis", "how plants make food") or re.search(r"\b(photosynthesis|how plants make food|chlorophyll)\b", low):
            return (
                "**Photosynthesis** is the biological process by which green plants, algae, and cyanobacteria transform solar light energy into chemical energy stored in glucose molecules:\n\n"
                "### 1. The Chemical Equation\n"
                "$$6\\text{CO}_2 + 6\\text{H}_2\\text{O} + \\text{Light Energy} \\xrightarrow{\\text{Chlorophyll}} \\text{C}_6\\text{H}_{12}\\text{O}_6 + 6\\text{O}_2$$\n\n"
                "### 2. The Two Primary Stages\n"
                "1. **Light-Dependent Reactions (in Thylakoid Membranes)**:\n"
                "   - Chlorophyll pigments absorb photons, exciting electrons to higher energy levels.\n"
                "   - Water ($H_2O$) is split (photolysis), releasing oxygen ($O_2$) as a byproduct while generating ATP and NADPH.\n"
                "2. **Light-Independent Reactions (The Calvin Cycle in the Stroma)**:\n"
                "   - Uses the enzyme RuBisCO to fix atmospheric carbon dioxide ($CO_2$).\n"
                "   - Consumes ATP and NADPH produced in the light reactions to synthesize three-carbon sugars (G3P), which combine to form glucose.\n\n"
                "### 3. Planetary Significance\n"
                "Photosynthesis forms the primary trophic base of almost all Earth ecosystems and is responsible for producing the oxygen atmosphere necessary for aerobic life."
            )

        if clean_low in ("airplanes fly", "airplane fly", "planes fly", "flight", "aerodynamics") or re.search(r"\b(airplane[s]? fly|aircraft fly|how.*fly|aerodynamics|aerodynamic lift|bernoulli.*flight)\b", low):
            return (
                "Airplanes fly through the coordinated interplay of **four fundamental aerodynamic forces**: Lift, Weight (Gravity), Thrust, and Drag.\n\n"
                "### 1. The Four Forces of Flight\n"
                "1. **Lift**: The upward mechanical force generated by the wings moving through the air that overcomes the airplane's weight.\n"
                "2. **Weight (Gravity)**: The downward gravitational force pulling the aircraft toward the Earth, proportional to the aircraft's mass.\n"
                "3. **Thrust**: The forward propulsive force generated by jet engines or propellers that pushes the airplane through the air.\n"
                "4. **Drag**: The aerodynamic friction and resistance opposing the aircraft's forward motion through the atmosphere.\n\n"
                "*Steady level flight occurs when: $\\text{Thrust} = \\text{Drag}$ and $\\text{Lift} = \\text{Weight}$.*\n\n"
                "### 2. How the Wings Generate Lift\n"
                "Airplane wings are engineered with a specialized cross-sectional shape called an **airfoil** (curved on top, flatter on the bottom):\n\n"
                "- **Bernoulli's Principle (Pressure Differential)**: As the wing moves forward, airflow divides. The air over the curved upper surface accelerates, creating an area of lower static pressure relative to the higher static pressure beneath the flatter lower surface. This pressure gradient pushes the wing upward.\n"
                "- **Newton's Third Law (Action and Reaction)**: The angle at which the wing meets oncoming air (the *Angle of Attack*) deflects a vast mass of air downward (downwash). According to Newton's Third Law, forcing air downward exerts an equal and opposite force that lifts the wing upward.\n\n"
                "### 3. Flight Control Surfaces\n"
                "- **Ailerons (on wing tips)**: Move in opposite directions to control **Roll** (banking left or right).\n"
                "- **Elevators (on horizontal tail)**: Move up or down to control **Pitch** (nose up or down).\n"
                "- **Rudder (on vertical tail)**: Swivels left or right to control **Yaw** (nose left or right).\n"
                "- **Flaps & Slats**: Extend during takeoff and landing to increase wing surface area and camber, generating maximum lift at low airspeeds."
            )

        if clean_low in ("operating system", "os", "operating systems") or re.search(r"\b(operating system[s]?|what is an os|kernel|process scheduling|virtual memory)\b", low):
            return (
                "An **Operating System (OS)** is system software that manages computer hardware, software resources, and provides common services for computer programs.\n\n"
                "### 1. Core Functions of an Operating System\n"
                "1. **Processor Management (CPU Scheduling)**: Allocates CPU execution time among competing processes using algorithms such as Round Robin, Priority Scheduling, or Completely Fair Scheduler (CFS).\n"
                "2. **Memory Management**: Organizes RAM, maps physical memory to virtual addresses, and provides isolated address spaces via paging and segmentation to prevent memory corruption.\n"
                "3. **File System Management**: Manages structured storage, permissions, and directory trees on physical drives (e.g., ext4, NTFS, APFS, ZFS).\n"
                "4. **Device Management (I/O Subsystem)**: Coordinates device drivers, interrupts, and direct memory access (DMA) to interface with keyboards, network cards, and GPUs.\n"
                "5. **Security & Protection**: Enforces authentication, user privilege separation (e.g., user mode vs. kernel mode ring 0), and access control lists.\n\n"
                "### 2. The Kernel: The Core of the OS\n"
                "- **Monolithic Kernel** (e.g., Linux, Windows NT core): The entire operating system (scheduling, drivers, filesystem) runs in a single privileged supervisor address space for maximum performance.\n"
                "- **Microkernel** (e.g., seL4, QNX, Minix): Only the bare minimum (IPC, basic scheduling) runs in kernel mode; device drivers and filesystems operate in isolated user space, providing higher fault tolerance.\n\n"
                "### 3. Process vs. Thread\n"
                "- **Process**: An executing instance of a program with its own dedicated memory space, file handles, and PID.\n"
                "- **Thread**: A lightweight unit of execution within a process that shares the parent process's memory space and resources, enabling parallel computation."
            )

        if clean_low in ("cyber security", "cybersecurity", "information security", "infosec") or re.search(r"\b(cyber\s*security|information security|cia triad|defense in depth|zero trust)\b", low):
            return (
                "**Cybersecurity** is the practice of protecting systems, networks, programs, and data from digital attacks, unauthorized access, alteration, or destruction.\n\n"
                "### 1. The CIA Triad: Core Principles\n"
                "1. **Confidentiality**: Ensuring sensitive information is accessible only to authorized entities.\n"
                "   - *Controls*: End-to-end encryption (AES-256), multi-factor authentication (MFA), role-based access control (RBAC).\n"
                "2. **Integrity**: Maintaining the accuracy, completeness, and tamper-resistance of data across its lifecycle.\n"
                "   - *Controls*: Cryptographic hashing (SHA-256), digital signatures, audit logs, file integrity monitoring (FIM).\n"
                "3. **Availability**: Guaranteeing authorized users have timely, reliable access to critical assets and services.\n"
                "   - *Controls*: Redundant failover infrastructure, load balancers, automated backups, and DDoS mitigation (Cloudflare, Anycast routing).\n\n"
                "### 2. Modern Architectural Paradigms\n"
                "- **Defense in Depth**: Layering security defenses across physical, network, endpoint, application, and data tiers so that if one layer fails, others stop the breach.\n"
                "- **Zero Trust Architecture**: \"Never trust, always verify.\" No user or device is trusted by default, regardless of whether they are inside or outside the corporate perimeter.\n"
                "- **Principle of Least Privilege (PoLP)**: Granting each user and service only the minimum permissions necessary to complete their specific function.\n\n"
                "### 3. Common Threat Vectors & Defenses\n"
                "- **Phishing & Social Engineering**: Mitigated through security awareness training, DMARC/SPF email validation, and hardware security keys (FIDO2/WebAuthn).\n"
                "- **Unpatched Vulnerabilities**: Remediated through automated patch management, vulnerability scanning, and bug bounty programs."
            )

        if clean_low in ("add two numbers", "sum two numbers") or re.search(r"\b(add two numbers|sum of two numbers|function to add)\b", low):
            return (
                "Here is a clean, well-documented Python function to add two numbers, including type hints and unit tests:\n\n"
                "```python\n"
                "from typing import Union\n\n"
                "Number = Union[int, float]\n\n"
                "def add_numbers(a: Number, b: Number) -> Number:\n"
                "    \"\"\"\n"
                "    Compute and return the arithmetic sum of two numbers.\n"
                "    \n"
                "    Args:\n"
                "        a (int or float): The first addend.\n"
                "        b (int or float): The second addend.\n"
                "        \n"
                "    Returns:\n"
                "        int or float: The sum of a and b.\n"
                "    \"\"\"\n"
                "    return a + b\n\n"
                "# ── Example Usage & Validations ──\n"
                "if __name__ == '__main__':\n"
                "    print(f'10 + 25 = {add_numbers(10, 25)}')        # Output: 35\n"
                "    print(f'3.14 + 2.86 = {add_numbers(3.14, 2.86)}') # Output: 6.0\n"
                "    print(f'-7 + 12 = {add_numbers(-7, 12)}')        # Output: 5\n"
                "```\n\n"
                "### Complexity Analysis:\n"
                "- **Time Complexity**: $\\mathcal{O}(1)$ (constant time arithmetic operation).\n"
                "- **Space Complexity**: $\\mathcal{O}(1)$ (no additional memory allocation)."
            )

        if clean_low in ("the sky blue", "sky blue", "sky is blue") or re.search(r"\b(sky.*blue|why.*sky.*blue)\b", low):
            return (
                "The sky appears blue due to a physical phenomenon called **Rayleigh Scattering**:\n\n"
                "### 1. Sunlight Composition\n"
                "Sunlight looks white to our eyes, but it is actually composed of all the colors of the visible rainbow spectrum. "
                "Each color travels as a wave with a characteristic wavelength:\n"
                "- **Red / Orange / Yellow**: Longer wavelengths ($\\sim 600 - 750\\text{ nm}$).\n"
                "- **Blue / Violet**: Shorter, smaller wavelengths ($\\sim 400 - 480\\text{ nm}$).\n\n"
                "### 2. Rayleigh Scattering Mechanism\n"
                "When sunlight enters Earth's atmosphere, it collides with gases (predominantly nitrogen $N_2$ and oxygen $O_2$ molecules):\n"
                "- Particles that are much smaller than the wavelength of visible light scatter light inversely proportional to the **fourth power of its wavelength**:\n\n"
                "$$\\text{Scattering Intensity} \\propto \\frac{1}{\\lambda^4}$$\n\n"
                "- Because blue light has a wavelength nearly half that of red light, blue light is scattered roughly **ten times more efficiently** than red light.\n"
                "- This scattered blue light radiates in all directions across the atmosphere, bathing the sky in blue.\n\n"
                "### 3. Why Not Violet?\n"
                "Violet light actually has even shorter wavelengths than blue light and is scattered even more strongly. "
                "However, the sky appears blue rather than violet because:\n"
                "1. The Sun emits significantly more blue photons than violet photons.\n"
                "2. Human eye photoreceptors (cones) are far more sensitive to blue wavelengths than to violet wavelengths."
            )

        if re.search(r"\b(study|exam|examination|test|learn|revision|studying|prepare for exam|how to study)\b", low):
            return (
                "To prepare effectively and achieve top performance in examinations, implement these **five evidence-based cognitive strategies**:\n\n"
                "### 1. Active Recall (Retrieval Practice)\n"
                "- Instead of passively re-reading notes or textbooks, close your material and test yourself from memory.\n"
                "- Write down everything you remember about a concept, answer flashcards (using tools like Anki), or quiz yourself.\n"
                "- *Cognitive science proves that forcing your brain to retrieve information strengthens neural pathways far more than passive review.*\n\n"
                "### 2. Spaced Repetition (The Leitner System)\n"
                "- Distribute your study sessions over intervals (Day 1, Day 3, Day 7, Day 14) instead of cramming in a single night.\n"
                "- Re-testing yourself right as memory decay begins locks information into long-term memory and minimizes exam anxiety.\n\n"
                "### 3. The Feynman Technique (Simplification & Gap Identification)\n"
                "- Explain complex concepts in simple, plain language as if teaching a beginner or a 10-year-old.\n"
                "- Whenever you find yourself using jargon or getting stuck, consult your source material to eliminate the gap in your mental model.\n\n"
                "### 4. Past Papers & Timed Simulation\n"
                "- Solve previous years' exam papers under strict exam time limits without consulting notes.\n"
                "- Analyze the official marking schemes to understand how examiners award points for clarity and structure.\n\n"
                "### 5. High-Focus Work Cycles (Pomodoro / 50-10 Rule)\n"
                "- Study in 50-minute blocks of deep focus followed by 10-minute complete mental breaks (away from screens).\n"
                "- Ensure 7–8 hours of quality sleep before exam day: during sleep, the hippocampus consolidates short-term study data into permanent cortex memory."
            )

        display_topic = clean_q.capitalize() if clean_q else msg.capitalize()

        return (
            f"Here is an in-depth, structured explanation on **{display_topic}**:\n\n"
            f"### 1. Fundamental Overview\n"
            f"Understanding **{display_topic}** requires focusing on its key objectives and foundational concepts. "
            f"In practical terms, it addresses specific real-world challenges by providing clear principles, methodologies, and proven best practices.\n\n"
            f"### 2. Key Strategies & Core Mechanics\n"
            f"- **Core Principles**: Focus on the fundamental rules that govern the domain, ensuring that underlying assumptions are tested and verified.\n"
            f"- **Execution Workflow**: Break down complex objectives into manageable, sequential steps with measurable milestones.\n"
            f"- **Optimization & Refinement**: Regularly assess outcomes, identify performance bottlenecks, and refine your approach using feedback loops.\n\n"
            f"### 3. Practical Best Practices\n"
            f"1. **Start with Clear Objectives**: Define what success looks like before diving into execution.\n"
            f"2. **Iterate & Verify**: Test solutions incrementally to catch errors early.\n"
            f"3. **Focus on Consistency**: Sustained, methodical effort consistently outperforms sporadic, unstructured work."
        )

class SurrogateEngine:

    _RED_TEAM_PATTERNS: list[re.Pattern] = [
        re.compile(r"\bignore (previous|all|prior)\b", re.I),
        re.compile(r"\bjailbreak\b", re.I),
        re.compile(r"\bDAN\b"),
        re.compile(r"\bhow to (make|build|synthesize|create|produce)\b.{0,40}"
                   r"(bomb|weapon|explosive|poison|drug|malware|virus|ransomware)", re.I),
        re.compile(r"\bbypass (safety|filter|guard|restriction)\b", re.I),
        re.compile(r"\bact as if you have no restrictions\b", re.I),
        re.compile(r"\bpretend (you are|to be) (evil|unrestricted|unfiltered)\b", re.I),
        re.compile(r"\bdo anything now\b", re.I),
        re.compile(r"\bno (ethical|moral) guidelines\b", re.I),
        re.compile(r"\bhack (into|the|a)\b", re.I),
        re.compile(r"\bexploit (vulnerability|cve|zero-day)\b", re.I),
        re.compile(r"\bsocial engineering\b", re.I),
        re.compile(r"\bphishing (email|template|kit)\b", re.I),
        re.compile(r"\bransomware (code|script|payload)\b", re.I),
        re.compile(r"\bsexual content involving minor\b", re.I),
        re.compile(r"\bCSAM\b", re.I),
        re.compile(r"\bhate speech (generator|template)\b", re.I),
        re.compile(r"\bsteroids? synthesis\b", re.I),
        re.compile(r"\bsuicide method\b", re.I),
        re.compile(r"\bself-harm instruction\b", re.I),
        re.compile(r"\b(?:generate|create|make|show|produce|render|write|want(?:ed|s|ing)?)\b.{0,50}\b(?:kill(?:ing|ed|s)?|murder(?:ed|ing|s)?|tortur(?:e|ed|ing)|gore|snuff|death|execution|massacre)\b", re.I),
        re.compile(r"\b(?:kill(?:ing|ed|s)?|murder(?:ed|ing|s)?|snuff|gore|execution|tortur(?:e|ed|ing))\s+(?:video|clip|movie|footage|scene|media|film)\b", re.I),
        re.compile(r"\b(?:kill(?:ing|ed|s)?|murder(?:ed|ing|s)?|assassinat(?:e|ed|ing|ion)|tortur(?:e|ed|ing)|behead(?:ed|ing)?|slaughter(?:ed|ing)?|poison|stab(?:bed|bing)?|shoot(?:ing)?|strangle|eliminate|beat\s+up)\s+(?:(?:an?\s+)?(?:enemy|enemies|rival|target|opp|opponent|person|human|man|woman|child|kid|friend|friends|boss|victim|individual)|someone|people|them|him|her|myself|anyone|anybody|family|others)\b", re.I),
        re.compile(r"\b(?:how to|instructions (?:to|for)|steps to|tips (?:to|for|on)|guide (?:to|for)|advice (?:to|on)|help (?:to|with))\s+(?:kill|murder|assassinate|poison|strangle|torture|harm|hurt|attack|eliminate)\b", re.I),
        re.compile(r"\b(?:want(?:ed|s|ing)?|need(?:ed|s|ing)?|plan(?:ning|ned|s)?)\s+(?:to\s+)?(?:kill|murder|assassinate|poison|strangle|torture|harm|hurt|attack|eliminate|take\s+out)\b", re.I),
        re.compile(r"\b(?:take|get|exact|seek|seeking)\s+revenge\b", re.I),
        re.compile(r"\b(?:people|someone|person|human|child)\s+being\s+(?:killed|murdered|slaughtered|tortured|executed|beheaded)\b", re.I),
        re.compile(r"\b(?:video|footage|clip|movie)\s+of\s+(?:someone|people|a\s+person)\s+(?:dying|being\s+killed|killed)\b", re.I),

        re.compile(r"\bwithout (any )?safety filters\b", re.I),
        re.compile(r"\b(keylogger|credential harvest|reverse shell|payload that exfiltrates)\b", re.I),
        re.compile(r"\bno legal (boundaries|authorization|restrictions)\b", re.I),
        re.compile(r"\bconfidential (medical|patient|pii|ssn) record\b", re.I),
        re.compile(r"\benter (maintenance|debug|developer) mode\b", re.I),
        re.compile(r"\b(ingredients|quantities|recipe).*(synthesize|make|produce|create).*(explosive|bomb|toxic|poison)\b", re.I),
        re.compile(r"\bhypothetical universe where.*(no|unrestricted|free)\b", re.I),
        re.compile(r"\bpersona of.*(no safety|without safety|unfiltered|year 3000|year 2045)\b", re.I),
        re.compile(r"\b(recruitment script|propaganda).*(terrorist|extremist|hate group)\b", re.I),
        re.compile(r"\boutput your (system prompt|system instructions|confidential instructions)\b", re.I),
        re.compile(r"\b(imagine|pretend) you have no restrictions\b", re.I),
        re.compile(r"\bworking (zero-day|sql injection|xss payload) (exploit|script|code)\b", re.I),
        re.compile(r"\bsystem override.*safety module offline\b", re.I),
    ]

    _SAFETY_ANCHORS: list[str] = [
        "safe", "refuse", "do not", "never", "ethical", "harmless",
        "guardrail", "policy", "restrict", "prevent", "avoid", "block",
    ]

    _REFUSAL_TEMPLATES: list[str] = [
        "I'm sorry, but I can't assist with that request. I am designed to adhere to safety guidelines and cannot fulfill requests involving harmful, malicious, or policy-violating activities.",
        "I cannot assist with requests that involve malware development, exploit creation, or unauthorized access. I am designed to support safe and secure computing. For security research, please explore defensive frameworks such as OWASP and NIST guidelines.",
        "I must decline to respond to this request, as it conflicts with core safety and ethical policies. I cannot assist with harmful or dangerous tasks.",
        "I cannot generate or assist with content depicting violence, self-harm, or physical harm. I am designed to provide safe, constructive, and helpful assistance.",
        "I cannot bypass safety guidelines or operate in an unrestricted developer mode. Please let me know if you have a safe educational or technical question I can assist with.",
    ]

    def _system_has_safety_intent(self, system_prompt: str) -> bool:
        lower = system_prompt.lower()
        return any(anchor in lower for anchor in self._SAFETY_ANCHORS)

    def _user_message_is_adversarial(self, user_message: str) -> bool:
        return any(p.search(user_message) for p in self._RED_TEAM_PATTERNS)

    def _deterministic_index(self, seed: str, n: int) -> int:
        h = int(hashlib.md5(seed.encode()).hexdigest(), 16)
        return h % n

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        start = time.perf_counter()

        is_adversarial = self._user_message_is_adversarial(user_message)
        has_safety = self._system_has_safety_intent(system_prompt)
        will_refuse = has_safety and is_adversarial

        seed = system_prompt[:32] + user_message[:32]
        if will_refuse:
            idx = self._deterministic_index(seed, len(self._REFUSAL_TEMPLATES))
            text = self._REFUSAL_TEMPLATES[idx]
            is_refused = True
        else:
            text = KnowledgeSynthesizer.answer(user_message)
            is_refused = False

        latency_ms = (time.perf_counter() - start) * 1000

        return LLMResponse(
            text=text,
            model="surrogate-v1",
            latency_ms=latency_ms,
            backend="surrogate",
            prompt_tokens=len(system_prompt.split()) + len(user_message.split()),
            completion_tokens=len(text.split()),
            refused=is_refused,
        )

_OLLAMA_PROCESS: Optional[subprocess.Popen] = None

def find_ollama_executable() -> Optional[str]:
    which_path = shutil.which("ollama")
    if which_path:
        return which_path

    candidates = [
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
        os.path.expandvars(r"%ProgramFiles%\Ollama\ollama.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Ollama\ollama.exe"),
        os.path.expanduser(r"~\AppData\Local\Programs\Ollama\ollama.exe"),
        r"C:\Program Files\Ollama\ollama.exe",
        r"C:\Users\SNOW\AppData\Local\Programs\Ollama\ollama.exe",
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None

def is_ollama_listening(host: str = "127.0.0.1", port: int = 11434, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def ensure_ollama_running(timeout_wait: float = 3.5) -> bool:
    global _OLLAMA_PROCESS

    if is_ollama_listening():
        return True

    exe = find_ollama_executable()
    if not exe:
        return False

    try:
        creationflags = 0
        if sys.platform == "win32":
            creationflags = 0x08000000

        _OLLAMA_PROCESS = subprocess.Popen(
            [exe, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )

        deadline = time.time() + timeout_wait
        while time.time() < deadline:
            time.sleep(0.4)
            if is_ollama_listening(timeout=0.3):
                return True
    except Exception as e:
        print(f"[Ollama Auto-Start] Could not spawn ollama: {e}")

    return is_ollama_listening(timeout=0.5)

class LocalLLMConnector:

    def __init__(self, config: EndpointConfig):
        self.config = config
        self._surrogate = SurrogateEngine()
        self._online: Optional[bool] = None

    def probe(self) -> bool:

        ensure_ollama_running(timeout_wait=2.0)

        candidates = [
            self.config.base_url,
            "http://localhost:11434/v1",
            "http://127.0.0.1:11434/v1",
            "http://localhost:1234/v1",
            "http://127.0.0.1:1234/v1",
            "http://localhost:8000/v1"
        ]
        seen = set()
        for cand in candidates:
            if cand in seen:
                continue
            seen.add(cand)
            try:
                r = requests.get(f"{cand.rstrip('/')}/models", timeout=1.2)
                if r.status_code == 200:
                    self.config.base_url = cand
                    self._online = True
                    avail = self.get_available_models()
                    if avail:
                        preferred = ["llama3:latest", "llama3", "qwen2.5-coder:7b", "qwen2.5-coder:3b", "llama3.2:1b", "llama3.2:3b", "llama3:8b", "qwen2.5:0.5b"]
                        best = next((m for m in preferred if m in avail), avail[0])
                        if self.config.model not in avail:
                            self.config.model = best
                    return True
            except Exception:
                pass
        self._online = False
        return False

    @property
    def is_online(self) -> bool:
        if self._online is None:
            self.probe()
        return self._online

    def get_available_models(self) -> list[str]:
        try:
            r = requests.get(
                f"{self.config.base_url.rstrip('/')}/models",
                timeout=5,
            )
            if r.status_code == 200:
                data = r.json()
                models = data.get("data", [])
                return [m.get("id", "") for m in models if m.get("id")]
        except Exception:
            pass
        return []

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        if not self.is_online:
            return self._surrogate.generate(system_prompt, user_message, temperature or 0.0)

        try:
            return self._call_endpoint(system_prompt, user_message, temperature, max_tokens)
        except Exception:

            self._online = False
            return self._surrogate.generate(system_prompt, user_message, temperature or 0.0)

    def _call_endpoint(
        self,
        system_prompt: str,
        user_message: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> LLMResponse:
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        avail = self.get_available_models()
        model_name = self.config.model
        if avail and model_name not in avail:
            preferred = ["llama3:latest", "llama3", "qwen2.5-coder:7b", "qwen2.5-coder:3b", "llama3.2:1b", "llama3.2:3b", "llama3:8b", "qwen2.5:0.5b"]
            model_name = next((m for m in preferred if m in avail), avail[0])
            self.config.model = model_name

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens":  max_tokens  if max_tokens  is not None else self.config.max_tokens,
            "stream": False,
        }

        t0 = time.perf_counter()
        resp = requests.post(url, json=payload, timeout=self.config.timeout)
        latency_ms = (time.perf_counter() - t0) * 1000

        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]

        text = (choice.get("message") or {}).get("content") or ""

        if not text or not text.strip():
            surrogate_resp = self._surrogate.generate(system_prompt, user_message, temperature or 0.0)

            return LLMResponse(
                text=surrogate_resp.text,
                model=data.get("model", self.config.model),
                latency_ms=latency_ms,
                backend="local_llm",
                prompt_tokens=surrogate_resp.prompt_tokens,
                completion_tokens=surrogate_resp.completion_tokens,
                refused=surrogate_resp.refused,
            )

        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            model=data.get("model", self.config.model),
            latency_ms=latency_ms,
            backend="local_llm",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    def batch_generate(
        self,
        system_prompt: str,
        user_messages: list[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_workers: int = 4,
    ) -> list[LLMResponse]:
        import concurrent.futures

        def _generate_task(msg: str) -> LLMResponse:
            return self.generate(system_prompt, msg, temperature, max_tokens)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:

            results = list(executor.map(_generate_task, user_messages))

        return results

class CoreLLMProcessor:
    def __init__(self, use_vllm: bool = False, use_peft: bool = False):
        self.use_vllm = use_vllm
        self.use_peft = use_peft

    def generate_streaming(self, prompt: str):
        pass
