**Multi-Head Attention（多头注意力）** 是 Transformer 模型的核心组件，用于捕捉输入序列中不同位置之间的复杂依赖关系。它通过并行多个独立的“注意力头”，让模型同时关注输入的不同子空间信息，显著提升了模型的表达能力。

---

### **核心思想**
注意力机制允许模型在处理每个位置的信息时，同时关注输入序列中的其他位置，从而捕捉长距离依赖关系。单头注意力机制虽然有效，但可能只能捕捉到一种模式的信息。这时候，Multi-Head Attention通过并行多个注意力头，每个头学习不同的注意力模式，然后将结果合并，提升模型的表达能力。

1. **单头注意力的问题**：
   - 传统注意力机制（如缩放点积注意力）只能学习一种模式的关联关系，例如局部依赖或全局依赖。
   - 单一注意力头可能无法充分捕捉输入中多样化的语义关联（如语法结构、语义角色等）。

2. **多头注意力的优势**：
   - 将输入拆分为多个子空间（头），每个头独立学习不同的注意力模式。
   - 最终合并所有头的输出，综合多种视角的信息，增强模型对复杂关系的建模能力。
   - 不同的头可以关注不同的子空间或不同的位置组合，比如有的头关注局部信息，有的关注全局依赖，或者不同语法结构的关系。这样模型能够更全面地理解输入序列的复杂关系。

---

### **数学实现**
3. **输入拆分**：
   - 输入向量通过线性变换生成多组 **Query（Q）、Key（K）、Value（V）** 矩阵。
   - 每个头的维度为 $d_{model}/h$（$h$ 是头数，$d_{model}$ 是输入维度）。

4. **单头注意力计算**：
   每个头独立计算 **缩放点积注意力**：
   
   $$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
   
   - $d_k$ 是 Key 的维度，缩放因子 $\sqrt{d_k}$ 用于防止点积过大导致梯度消失。

5. **多头合并**：
   - 所有头的输出拼接后，通过线性变换还原为原始维度：
   
   $$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O $$
   
   - $W^O$ 是输出权重矩阵。

---

### **直观理解**
- **示例**：在翻译句子“The cat sat on the mat”时：
  - **头1** 可能关注“cat”与“sat”之间的主谓关系。
  - **头2** 可能捕捉“on”与“mat”之间的介词-宾语关系。
  - **头3** 可能关联“The”与“cat”的冠词-名词关系。
- 最终模型综合所有头的注意力结果，生成更准确的表示。

---

### **在 Transformer 中的应用**
6. **编码器**：
   - 自注意力（Self-Attention）：输入序列内部各位置间的关联。
   - 每个词通过多头注意力整合上下文信息（如“bank”在“river bank”和“bank account”中的不同含义）。

7. **解码器**：
   - 自注意力 + 交叉注意力（Cross-Attention）：解码时关注编码器的输出。
   - 例如，生成翻译时，解码器通过交叉注意力聚焦源语言的关键词。

---

### **为什么有效？**
8. **多样化表征**：
   - 不同头学习不同的注意力模式（如局部/全局、语法/语义）。
   - 类似卷积神经网络中多通道滤波器的思想。

9. **并行计算**：
   - 各头独立计算，可通过 GPU 并行加速。

10. **鲁棒性**：
   - 避免单一注意力头对噪声敏感的问题。

---

### **代码示例（PyTorch）**
```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, h=8):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        
        # 线性变换矩阵
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 拆分多头
        Q = self.W_Q(Q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        
        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_O(output)
```

---

### **实际应用中的优化**
- **稀疏注意力**：减少计算量（如 Longformer、BigBird）。
- **FlashAttention**：通过硬件优化加速计算。
- **跨头参数共享**：降低模型复杂度（如 ALBERT）。

理解 Multi-Head Attention 是掌握 Transformer、BERT、GPT 等现代大模型的关键一步。