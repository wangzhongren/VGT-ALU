# VGT-Pro: Neural Arithmetic Logic Unit with Dilated Iterative Convolution

VGT-Pro is a neural arithmetic logic unit (ALU) that leverages a **dilated iterative convolutional architecture** to perform symbolic-numeric hybrid computation. Trained solely on 1–6 digit addition, it achieves **100% accuracy up to 20-digit extrapolation**, demonstrating emergent algorithmic reasoning through geometric pressure and dynamic receptive field expansion.

---

## 🧠 Core Architecture

### `VGTProModel` – The Semantic Arithmetic Kernel
- **Input**: Two reversed-digit sequences (e.g., `[3,2,1] + [6,5,4]` for `123 + 456`)
- **Embedding**: Each digit → 128-dim vector
- **Feature Fusion**: Concatenates operand embeddings, reduced via 1×1 conv
- **Iterative Processing**:
  - **Dynamic dilation**: Cycles through dilation rates `[1, 2, 4]` over iterations
  - **Residual propagation**: Ensures long-range carry signals propagate fully
  - **Redundant steps**: `seq_len + 4` iterations guarantee convergence
- **Output**: Per-digit logits → decoded into final number

> This design mimics a **neural finite-state machine** where each convolutional layer acts as a logic gate with adaptive memory span.

---

## ⚙️ Functional Modules

### 1. `vgt_alu_core.py` – Neural ALU Core
Implements a full arithmetic instruction set using only the trained `add` primitive:
- **`add(a, b)`**: Native operation via forward pass
- **`sub(a, b)`**: Implemented via **10's complement arithmetic**
- **`mul(a, b)`**: Recursive digit-wise accumulation (`a × b = Σ a × digit_i × 10^i`)
- **`compare(a, b)`**: Leverages `sub` to determine ordering

> All operations are **symbolically grounded**—no floating-point approximation.

### 2. `vgt_shell.py` – Interactive Shell
Provides a REPL interface for real-time arithmetic:
```bash
>>> 12345678901234567890+98765432109876543210
Result: 111111111011111111100
```
Supports `+`, `-`, `*` with arbitrary-length integers.

### 3. `train_core.py` – Training Engine
Key innovations:
- **Geometric Collapse Loss**: 
  ```python
  loss = CE + α(t) · ||h||₂
  ```
  - `α(t)` follows an **arch-shaped annealing schedule** (ramps up to 50, then decays)
  - Forces internal states to polarize into discrete logic pathways
- **Dynamic Digit Mixing**: Randomly samples 1–6 digit problems per batch
- **Extrapolation Validation**: Tests up to 20 digits during training

---

## 📈 Performance

| Digits | Accuracy |
|--------|----------|
| 1–6    | 100%     |
| 12     | 100%     |
| 16     | 100%     |
| 20     | 100%     |

> Model file: `vgt_pro_logic_machine.pth`  
> Metadata: `vgt_pro_logic_machine_meta.json`

---

## ▶️ Quick Start

1. **Train the model**:
   ```bash
   python train_core.py
   ```

2. **Launch interactive shell**:
   ```bash
   python vgt_shell.py
   ```

3. **Use core ALU in code**:
   ```python
   from vgt_alu_core import NeuralALU
   alu = NeuralALU("vgt_pro_logic_machine.pth")
   print(alu.mul(1234, 5678))  # → 7006652
   ```

---

## 🔬 Design Philosophy

VGT-Pro demonstrates that **algorithmic generalization** can emerge from:
1. **Structured inductive bias** (reversed digits + convolutional recurrence)
2. **Geometric regularization** (L2 pressure on hidden states)
3. **Temporal redundancy** (over-iterated processing)

This system blurs the line between neural networks and symbolic machines—each inference step is a **deterministic classification** over digit states, not a probabilistic guess.

> **Note**: The model assumes little-endian digit order internally. All I/O uses standard big-endian notation.

---

*VGT-Pro: Where every convolution is a carry, and every residual is a proof.*