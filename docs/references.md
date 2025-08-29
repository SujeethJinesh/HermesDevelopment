# HERMES References and Citations

## Core Technologies

### HuggingFace Transformers
- **Transformers Library**: Wolf, T., et al. (2020). "Transformers: State-of-the-Art Natural Language Processing." Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations. [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)

### Quantization

#### BitsAndBytes
- **bitsandbytes Documentation**: Dettmers, T. (2023). "8-bit and 4-bit Quantization with bitsandbytes." [https://huggingface.co/docs/bitsandbytes](https://huggingface.co/docs/bitsandbytes)
- **CUDA-only Limitation**: [https://github.com/TimDettmers/bitsandbytes#requirements](https://github.com/TimDettmers/bitsandbytes#requirements)
- **Apple Silicon Discussion**: [https://github.com/TimDettmers/bitsandbytes/issues/30](https://github.com/TimDettmers/bitsandbytes/issues/30)

#### Alternative Quantization Formats
- **GGUF Format**: llama.cpp team. "GGUF - GPT-Generated Unified Format." [https://github.com/ggerganov/llama.cpp/blob/master/docs/gguf.md](https://github.com/ggerganov/llama.cpp/blob/master/docs/gguf.md)
- **MLX**: Apple Machine Learning Research. "MLX: An array framework for Apple silicon." [https://github.com/ml-explore/mlx](https://github.com/ml-explore/mlx)

### Offline Mode and Hermetic Execution

#### HuggingFace Hub Offline Mode
- **Offline Environment Variables**: [https://huggingface.co/docs/huggingface_hub/guides/download#download-files-to-local-folder](https://huggingface.co/docs/huggingface_hub/guides/download#download-files-to-local-folder)
- **Local Files Only**: [https://huggingface.co/docs/transformers/installation#offline-mode](https://huggingface.co/docs/transformers/installation#offline-mode)

### Concurrency and Process Management

#### Python Multiprocessing
- **ProcessPoolExecutor**: Python Software Foundation. "concurrent.futures — Launching parallel tasks." [https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor](https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor)
- **Spawn Context**: [https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods)

### Device Management

#### CUDA
- **PyTorch CUDA Semantics**: [https://pytorch.org/docs/stable/notes/cuda.html](https://pytorch.org/docs/stable/notes/cuda.html)
- **Accelerate Library**: HuggingFace. "Accelerate: Training and inference at scale made simple." [https://huggingface.co/docs/accelerate](https://huggingface.co/docs/accelerate)

#### Apple Silicon
- **MPS Backend**: Apple. "Accelerated PyTorch training on Mac." [https://developer.apple.com/metal/pytorch/](https://developer.apple.com/metal/pytorch/)
- **Unified Memory Architecture**: [https://developer.apple.com/documentation/metal/gpu_devices_and_work_submission/gpu_device_data](https://developer.apple.com/documentation/metal/gpu_devices_and_work_submission/gpu_device_data)

### Model Serving Backends

#### Ollama
- **Concurrency Limitations**: Ollama Documentation. "Known Limitations." [https://github.com/ollama/ollama/blob/main/docs/faq.md#concurrency](https://github.com/ollama/ollama/blob/main/docs/faq.md#concurrency)

#### vLLM
- **vLLM Paper**: Kwon, W., et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP '23. [https://vllm.ai/](https://vllm.ai/)

### Benchmarks and Datasets

#### SWE-bench
- **SWE-bench Paper**: Jimenez, C.E., et al. (2024). "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" ICLR 2024. [https://www.swebench.com/](https://www.swebench.com/)
- **SWE-bench Lite**: [https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite)

### Serialization and Transport

#### Protocol Buffers
- **Protobuf Documentation**: Google. "Protocol Buffers." [https://protobuf.dev/](https://protobuf.dev/)

#### gRPC
- **gRPC Documentation**: "A high performance, open source universal RPC framework." [https://grpc.io/](https://grpc.io/)

### Storage and Caching

#### Content-Addressed Storage
- **MCP (Message Content Protocol)**: Internal HERMES design for content-addressed storage with TTLs.

### Testing and Reproducibility

#### Hermetic Testing
- **Bazel Hermetic Testing**: Google. "Hermeticity." [https://bazel.build/basics/hermeticity](https://bazel.build/basics/hermeticity)

#### Deterministic Execution
- **PyTorch Reproducibility**: [https://pytorch.org/docs/stable/notes/randomness.html](https://pytorch.org/docs/stable/notes/randomness.html)

## Performance Benchmarks

### Token Generation Speeds

#### Apple Silicon (MPS)
- Typical speeds with quantized models: 15-50 tokens/second depending on model size
- Source: Community benchmarks and internal testing

#### NVIDIA H100
- Production speeds with vLLM: 100-500+ tokens/second with batching
- Source: vLLM performance benchmarks

## Platform-Specific Notes

### CUDA Platform
- Supports full bitsandbytes quantization (4-bit and 8-bit)
- Best performance with H100/A100 GPUs
- Memory optimization through PagedAttention (vLLM)

### Apple Silicon Platform  
- No bitsandbytes support (CUDA-only library)
- Alternative: Use smaller models (≤8B parameters)
- Future: MLX or GGUF quantization formats
- Unified memory architecture allows efficient CPU-GPU data sharing

### CPU Platform
- Fallback mode with no acceleration
- Limited to float16/float32 precision
- Suitable for development and testing only

## Security and Isolation

### Process Isolation
- Each model instance runs in separate OS process
- No shared memory between processes
- Clean teardown on process termination

### Offline Execution
- All models pre-cached locally
- Network access blocked during hermetic evaluation
- Deterministic results with fixed seeds

## Related Work

### Multi-Agent Systems
- AutoGen: Microsoft. "AutoGen: Enabling Next-Gen LLM Applications." [https://github.com/microsoft/autogen](https://github.com/microsoft/autogen)
- LangChain: "Building applications with LLMs." [https://langchain.com/](https://langchain.com/)

### LLM Optimization
- FlashAttention: Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness."
- Paged Attention: Used in vLLM for efficient memory management

## Implementation Resources

### Code Examples
- HuggingFace Model Parallelism: [https://huggingface.co/docs/transformers/parallelism](https://huggingface.co/docs/transformers/parallelism)
- PyTorch Distributed: [https://pytorch.org/tutorials/intermediate/ddp_tutorial.html](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

### Configuration Management
- YAML Configuration: [https://yaml.org/spec/1.2.2/](https://yaml.org/spec/1.2.2/)
- Environment Variables Best Practices: [https://12factor.net/config](https://12factor.net/config)