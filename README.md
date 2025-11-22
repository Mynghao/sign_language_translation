# **Sign Language Translation (SLT) — ST-GCN + Seq2Seq**
*A full pipeline for translating sign language videos into natural language text and speech.*

This project builds an end-to-end system for **automatic sign language translation** using **skeleton-based motion capture**, **graph neural networks**, and **sequence modeling**.  
The pipeline processes OpenPose pose-keypoint data extracted from sign-language videos, encodes spatiotemporal joint dynamics using an **ST-GCN**, and decodes the resulting embeddings into text (and optionally synthesizes speech).

---

## ✨ Key Features

### 🧍‍♂️ Skeleton-Based Processing  
- Uses **OpenPose** JSON keypoint data  
- Tracks body, face, and both hands  
- Works across multi-view camera setups  
- Handles variable-length sequences

### 🧠 Spatio-Temporal Graph Convolutional Network (ST-GCN) Encoder  
- Nodes = human joints  
- Edges = anatomical connections  
- Learns motion + structure jointly  
- Outputs sequence embeddings for decoding

### 🔁 Sequence Decoder  
Choose either:
- **RNN (GRU/LSTM)** for lightweight decoding  
- **Transformer** for richer long-range dependencies  

Converts motion embeddings into **English sentences**.

### 🔊 Optional Speech Output  
Final text is passed into a lightweight TTS model for audio synthesis.
