# keep_gpu_busy.py
import torch

device = "cuda:0"

while True:
    # 20000 x 20000 float32 행렬 생성 (~1.6GB 메모리)
    x = torch.randn(20000, 20000, device=device)
    # 행렬 곱 (이 부분에서 GPU 연산 풀로드)
    y = x @ x
    # 결과를 살짝 사용해 주면 PyTorch가 최적화로 연산을 생략하지 않음
    _ = y.sum().item()
