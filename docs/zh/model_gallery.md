# 模型画廊 (Model Gallery)

本页展示了使用 `torchview` 生成的经典深度学习模型可视化结果。

所有图片均为 **SVG 矢量图**，您可以**点击图片**或在浏览器中**放大**以查看所有细节。

---

## 🖼️ 计算机视觉 (Computer Vision)

### ResNet-50
![ResNet-50](images/ResNet50.svg)

**残差网络**：经典的 CNN 架构，通过残差连接（Skip Connections）解决了深层网络的训练难题。图中清晰展示了 Bottleneck 结构。

### Inception V3
![Inception V3](images/InceptionV3.svg)

**Inception 架构**：以多尺度分支（Inception Module）著称，结构复杂但高效。

### Vision Transformer (ViT)
![ViT](images/ViT_B_16.svg)

**ViT**：将 Transformer 架构直接应用于图像块（Patches），完全摒弃了卷积操作。

### FCN (Fully Convolutional Network)
![FCN](images/FCN_ResNet50.svg)

**全卷积网络**：用于语义分割的经典模型，基于 ResNet-50 主干。

---

## 📝 自然语言处理 (NLP)

### Transformer Encoder (BERT-style)
![Transformer Encoder](images/Transformer_Encoder.svg)

**Transformer 编码器**：包含 Multi-Head Attention 和 Feed Forward Network，是 BERT 等模型的基础。

### Seq2Seq with Attention
![Seq2Seq](images/Seq2Seq_Attention.svg)

**序列到序列 + 注意力机制**：经典的机器翻译架构。图中展示了 Encoder（左）和 Decoder（右）以及中间的 Attention 计算过程（横向布局）。

---

## 🛒 推荐系统 (Recommender Systems)

### Wide & Deep
![Wide & Deep](images/Wide_and_Deep.svg)

**Wide & Deep**：结合了线性模型（Wide，记忆能力）和深度神经网络（Deep，泛化能力）的经典推荐架构。

---

## 🎨 生成模型 (Generative Models)

### VAE (Variational Autoencoder)
![VAE](images/VAE.svg)

**变分自编码器**：图中展示了完整的生成流程：Encoder → Reparameterization (采样) → Decoder。

---

## ⭐ 特殊架构 (Special Architectures)

### WaveNet
![WaveNet](images/WaveNet.svg)

**WaveNet**：用于语音合成的生成模型。图中清晰展示了基于空洞卷积（Dilated Convolution）的残差块堆叠结构（横向布局）。

### GCN (Graph Convolutional Network)
![GCN](images/GCN.svg)

**图卷积网络**：处理图结构数据的神经网络。

---

## 如何贡献

如果您想为画廊添加新的模型，请确保：
1. 模型结构具有代表性。
2. 使用 `torchview` 生成 SVG 格式以保证清晰度。
3. 提交 PR 到我们的仓库。
