# keep_gpu_busy_mp.py
import torch
from torch.multiprocessing import Process

def worker(dev: int):
    torch.cuda.set_device(dev)
    device = f"cuda:{dev}"
    m = 20000  # 행렬 한 변 길이 (필요하면 키워도 됨)
    while True:
        x = torch.randn(m, m, device=device)  # 생성 자체도 GPU 연산
        y = x @ x                              # 큰 matmul로 풀로드
        _ = y.sum().item()                     # 결과 사용해서 최적화 방지

if __name__ == "__main__":
    procs = []
    for d in (0, 1):               # gpu:0, gpu:1 각각에 프로세스 하나씩
        p = Process(target=worker, args=(d,), daemon=False)
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
