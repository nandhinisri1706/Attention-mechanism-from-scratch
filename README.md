# Self-Attention Mechanism using NumPy

## 📌 Project Overview

This project demonstrates the implementation of the **Self-Attention Mechanism** from scratch using **Python** and **NumPy** without relying on deep learning frameworks like TensorFlow or PyTorch.

Self-Attention is a core component of the **Transformer architecture**, allowing each input token to focus on the most relevant tokens while processing information.

---

## 🎯 Objective

- Understand the working of the Self-Attention mechanism.
- Implement each mathematical step manually.
- Learn how Queries (Q), Keys (K), and Values (V) are generated.
- Calculate attention scores and final attention output.

---

## 🛠️ Technologies Used

- Python 3
- NumPy

---

## 📂 Project Structure

```
Self-Attention/
│── self_attention.py
│── README.md
```

---

## ⚙️ Working Steps

### Step 1: Input Embeddings
Create sample input embeddings representing three tokens.

### Step 2: Initialize Weight Matrices
Randomly initialize:

- Query Weight Matrix (WQ)
- Key Weight Matrix (WK)
- Value Weight Matrix (WV)

### Step 3: Generate Q, K, and V

Using matrix multiplication:

```
Q = X × WQ
K = X × WK
V = X × WV
```

### Step 4: Calculate Attention Scores

```
Scores = Q × Kᵀ
```

This measures the similarity between every Query and Key.

### Step 5: Scale the Scores

```
Scaled Scores = Scores / √dₖ
```

Scaling prevents extremely large values before applying Softmax.

### Step 6: Apply Softmax

Softmax converts the scaled scores into probability values called **Attention Weights**.

```
Attention Weights = Softmax(Scaled Scores)
```

### Step 7: Compute Final Output

```
Attention Output = Attention Weights × V
```

This produces the final context-aware representation.

---

## 📊 Output

The program displays:

- Input Embeddings
- Query Matrix (Q)
- Key Matrix (K)
- Value Matrix (V)
- Attention Scores
- Scaled Scores
- Attention Weights
- Final Attention Output

---

## 📖 Concepts Covered

- Self-Attention
- Transformer Architecture
- Query, Key, Value (QKV)
- Matrix Multiplication
- Softmax Function
- Scaled Dot-Product Attention
- NumPy Operations

---

## 🚀 How to Run

Clone the repository:

```bash
git clone https://github.com/your-username/self-attention-mechanism.git
```

Move into the project folder:

```bash
cd self-attention-mechanism
```

Install NumPy:

```bash
pip install numpy
```

Run the program:

```bash
python self_attention.py
```

---

## 📚 Learning Outcome

After completing this project, you will understand:

- How Self-Attention works internally.
- How Transformers process input sequences.
- The role of Queries, Keys, and Values.
- How attention weights are calculated using Softmax.
- How context-aware representations are generated.

---

## 👩‍💻 Author

**Nandhini Sri**

## ⭐ Future Improvements

- Multi-Head Attention
- Positional Encoding
- Transformer Encoder Layer
- Visualization of Attention Weights
- PyTorch Implementation
