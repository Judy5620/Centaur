# keep_gpu_busy_multi.py
import torch

devices = ["cuda:0", "cuda:1"]

while True:
    for device in devices:
        # 20000 x 20000 float32 행렬 생성 (~1.6GB 메모리)
        x = torch.randn(20000, 20000, device=device)
        # 행렬 곱 (GPU 연산 풀로드)
        y = x @ x
        # 결과를 강제로 사용 (연산 최적화 방지)
        _ = y.sum().item()
