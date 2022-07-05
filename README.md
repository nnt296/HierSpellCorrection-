# Unofficial implementation of Hierarchical Transformer Encoders for Vietnamese Spelling Correction

## Dependencies
- Transformer
- Pytorch
- nltk
- Pytorch-lightning

## Training
- Train WordLevelTokenizer:
  ```
  # Prepare corpus file and run
  python -m models.word_char_tokenizer
  ```
- Prepare model:
  ```
  python -m models.baseline
  ```
- Train model:
  ```
  # Change params and run main
  python main.py
  ```

- Distributed training:
  ```
  # https://pytorch-lightning.readthedocs.io/en/latest/clouds/cluster_intermediate_1.html
  export MASTER_PORT=...
  export MASTER_ADDR=...
  export WORLD_SIZE=2
  export NODE_RANK=1
  
  # Select port and enable debugging
  NCCL_SOCKET_IFNAME="enp3s0" NCCL_DEBUG=INFO python main.py
  ```


## References
```
@misc{https://doi.org/10.48550/arxiv.2105.13578,
  doi = {10.48550/ARXIV.2105.13578},
  url = {https://arxiv.org/abs/2105.13578},
  author = {Tran, Hieu and Dinh, Cuong V. and Phan, Long and Nguyen, Son T.},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Hierarchical Transformer Encoders for Vietnamese Spelling Correction},
  publisher = {arXiv},
  year = {2021},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
