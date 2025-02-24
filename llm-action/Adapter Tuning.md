
### **Adapter Tuning 及其变体详解**

Adapter Tuning 是一种**参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）**方法，旨在通过插入少量可训练模块（Adapter）到预训练模型中，仅微调这些模块而非整个模型，从而降低计算和存储成本。以下是其核心原理及主要变体：

---

### **1. 原始 Adapter Tuning**
#### **核心思想**
- 在预训练模型的每一层（如Transformer层）中插入小型神经网络模块（Adapter），仅训练这些模块和部分参数（如LayerNorm），冻结原始模型参数。
- **优势**：大幅减少可训练参数量（通常仅占全量微调的1-10%），适用于多任务学习和资源受限场景。

#### **结构设计**
- **典型结构**：  
  $$
  \text{Adapter}(x) = x + W_{\text{up}} \cdot \text{ReLU}(W_{\text{down}} \cdot x)
  $$
  - $W_{\text{down}} \in \mathbb{R}^{d \times m}$（降维，如 $m = 64$）
  - $W_{\text{up}} \in \mathbb{R}^{m \times d}$（升维）
  - **瓶颈设计**：通过降维-激活-升维减少参数量。
- **插入位置**：通常放在Transformer层的自注意力模块和前馈网络（FFN）之后。

---

### **2. 主要变体及改进**

#### **(1) AdapterFusion**
- **目标**：结合多个任务特定的Adapter，提升多任务学习效率。
- **方法**：
  - 为每个任务训练独立的Adapter。
  - 新增一个**注意力层**，动态学习如何融合不同Adapter的输出。
- **优点**：避免任务干扰，实现知识共享。

#### **(2) Compacter / Low-Rank Adapters**
- **目标**：进一步减少Adapter参数。
- **方法**：
  - **参数化超复杂乘法（PHM）**：将权重矩阵分解为低秩矩阵乘积。
  - **共享参数**：跨层或跨任务共享部分参数。
- **参数量**：可降至原始Adapter的1/10。

#### **(3) Parallel Adapter**
- **改进点**：将Adapter与原有层**并行**而非串联。
  $$
  \text{Output} = \text{FFN}(x) + \text{Adapter}(x)
  $$
- **优点**：减少串行结构带来的推理延迟。

#### **(4) MAM Adapter (Mix-and-Match)**
- **组合技术**：集成Adapter与**LoRA（Low-Rank Adaptation）**。
  - **LoRA**：通过低秩矩阵更新原始权重（$W = W_0 + \Delta W$，其中 $\Delta W = A \cdot B$, $A \in \mathbb{R}^{d \times r}, B \in \mathbb{R}^{r \times d}$, $r \ll d$）。
  - **分工**：Adapter处理层间变换，LoRA调整注意力权重。
- **优点**：综合参数效率和性能。

#### **(5) Scaled Parallel Adapter**
- **改进点**：在并行Adapter中引入可学习的**缩放因子**：
  $$
  \text{Output} = \text{FFN}(x) + \lambda \cdot \text{Adapter}(x)
  $$
  -  $\lambda$ 可动态调整Adapter的影响强度。
- **优点**：增强灵活性，适应不同任务难度。

#### **(6) Prefix-Tuning**
- **关联技术**：虽非Adapter变体，但同属PEFT方法。
- **方法**：在输入序列前添加可学习的**前缀向量（Prefix）**，引导模型生成任务相关输出。
- **特点**：无需修改模型结构，仅调整输入表示。

---

### **3. 方法对比**

| 方法               | 参数量     | 训练速度 | 任务表现 | 适用场景               |
|--------------------|------------|----------|----------|------------------------|
| **原始Adapter**    | 中（~1%）  | 中等     | 高       | 单任务/多任务          |
| **AdapterFusion**  | 高（~5%）  | 慢       | 更高     | 多任务知识融合         |
| **Compacter**      | 极低（~0.1%）| 快     | 中等     | 资源严格受限           |
| **MAM Adapter**    | 低（~0.5%）| 快       | 高       | 平衡效率与性能         |
| **Prefix-Tuning**  | 极低（~0.1%）| 快     | 中等     | 生成任务/提示工程      |

---

### **4. 实验结论**
- **参数效率**：Adapter变体通常仅需1%的参数量即可达到全量微调90%以上的性能。
- **多任务学习**：AdapterFusion在跨任务泛化中表现突出。
- **推理延迟**：Parallel Adapter比串行结构更高效，适合实时应用。

---

### **5. 代码示例（PyTorch）**
```python
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, d_model, bottleneck_dim):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        return x + self.up(self.activation(self.down(x)))

# 在Transformer层中插入Adapter
class TransformerLayerWithAdapter(nn.Module):
    def __init__(self, original_layer, bottleneck_dim=64):
        super().__init__()
        self.original_layer = original_layer  # 原始Transformer层
        self.adapter = Adapter(original_layer.d_model, bottleneck_dim)
        
    def forward(self, x):
        x = self.original_layer(x)  # 原始层计算
        x = self.adapter(x)         # Adapter微调
        return x
```

---

### **总结**
Adapter Tuning 及其变体通过模块化设计，在参数效率与模型性能之间取得平衡，成为大模型微调的主流方法。选择具体变体时需权衡任务需求、资源限制和性能目标，例如：
- **资源受限**：选用Compacter或Prefix-Tuning。
- **多任务场景**：AdapterFusion更优。
- **低延迟需求**：Parallel Adapter或MAM Adapter是理想选择。