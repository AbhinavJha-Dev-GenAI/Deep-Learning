# Seq2Seq Architectures (Encoder-Decoder)

**Seq2Seq (Sequence-to-Sequence)** models are designed to turn one sequence into another (e.g., translating English to French).

## The Core Architecture

### 1. Encoder
- Processes the input sequence and compresses it into a single **Context Vector** (also called Thought Vector).
- This vector aims to capture the "meaning" of the entire input.

### 2. Decoder
- Takes the context vector as its initial hidden state.
- Predicts the output sequence step-by-step until an `<EOS>` (End Of Sentence) token is reached.

## Landmark Innovations

### Attention Mechanism
The main weakness of standard Seq2Seq is the bottleneck caused by the fixed-length context vector. **Attention** allows the decoder to "look back" at specific parts of the input sequence at each step.

### Beam Search
Instead of just picking the most likely word at each step (Greedy Search), **Beam Search** keeps track of multiple likely candidates to find a globally better sequence.

## Use Cases
- Machine Translation.
- Text Summarization.
- Image Captioning (Image -> Sequence).

## Implementation Note
Modern Seq2Seq tasks are now mostly handled by **Transformers**, but understanding the Encoder-Decoder RNN structure is vital for learning how we got here.
