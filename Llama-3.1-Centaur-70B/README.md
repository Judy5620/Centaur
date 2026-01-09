# A foundation model to predict and capture human cognition


<img src="https://marcelbinz.github.io/imgs/centaur.png" width="200"/>

- **Paper:** [A foundation model to predict and capture human cognition](https://www.nature.com/articles/s41586-025-09215-4)
- **Point of Contact:** [Marcel Binz](mailto:marcel.binz@helmholtz-munich.de)

Establishing a unified theory of cognition has been a major goal of psychology. While there have been previous attempts to instantiate such theories by building computational models, we currently do not have one model that captures the human mind in its entirety. A first step in this direction is to create a model that can predict human behavior in a wide range of settings. Here we introduce Centaur, a computational model that can predict and simulate human behavior in any experiment expressible in natural language. We derived Centaur by finetuning a state-of-the-art language model on a novel, large-scale data set called Psych-101. Psych-101 reaches an unprecedented scale, covering trial-by-trial data from over 60,000 participants performing over 10,000,000 choices in 160 experiments. Centaur not only captures the behavior of held-out participants better than existing cognitive models, but also generalizes to new cover stories, structural task modifications, and entirely new domains. Furthermore, we find that the model's internal representations become more aligned with human neural activity after finetuning. Taken together, our results demonstrate that it is possible to discover computational models that capture human behavior across a wide range of domains. We believe that such models provide tremendous potential for guiding the development of cognitive theories and present a case study to demonstrate this. 

## Usage

Note that Centaur is trained on a data set in which human choices are encapsuled by "<<" and ">>" tokens. For optimal performance, it is recommended to adjust prompts accordingly.

The recommended usage is by loading the low-rank adapter using unsloth:

```python
from unsloth import FastLanguageModel

model_name = "marcelbinz/Llama-3.1-Centaur-70B-adapter"
model, tokenizer = FastLanguageModel.from_pretrained(
  model_name = model_name,
  max_seq_length = 32768,
  dtype = None,
  load_in_4bit = True,
)

FastLanguageModel.for_inference(model)
```

This requires 80 GB GPU memory. [test_adapter.py](https://github.com/marcelbinz/Llama-3.1-Centaur-70B/blob/main/test_adapter.py) shows an example of this type of usage.

You can alternatively use the model with the HuggingFace Transformers library:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "marcelbinz/Llama-3.1-Centaur-70B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

This requires at least 160 GB GPU memory (even more for longer prompts). [test.py](https://github.com/marcelbinz/Llama-3.1-Centaur-70B/blob/main/test.py) shows an example of this type of usage.


### Reference

```
@misc{binz2024centaurfoundationmodelhuman,
      title={Centaur: a foundation model of human cognition}, 
      author={Marcel Binz and Elif Akata and Matthias Bethge and Franziska Brändle and Fred Callaway and Julian Coda-Forno and Peter Dayan and Can Demircan and Maria K. Eckstein and Noémi Éltető and Thomas L. Griffiths and Susanne Haridi and Akshay K. Jagadish and Li Ji-An and Alexander Kipnis and Sreejan Kumar and Tobias Ludwig and Marvin Mathony and Marcelo Mattar and Alireza Modirshanechi and Surabhi S. Nath and Joshua C. Peterson and Milena Rmus and Evan M. Russek and Tankred Saanum and Natalia Scharfenberg and Johannes A. Schubert and Luca M. Schulze Buschoff and Nishad Singhi and Xin Sui and Mirko Thalmann and Fabian Theis and Vuong Truong and Vishaal Udandarao and Konstantinos Voudouris and Robert Wilson and Kristin Witte and Shuchen Wu and Dirk Wulff and Huadong Xiong and Eric Schulz},
      year={2024},
      eprint={2410.20268},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.20268}, 
}
```

### Dependencies

The code was tested using the following packages and versions.

```
_libgcc_mutex=0.1=main
_openmp_mutex=5.1=1_gnu
accelerate=0.30.1=pypi_0
aiohttp=3.9.5=pypi_0
aiosignal=1.3.1=pypi_0
asttokens=2.4.1=pypi_0
async-timeout=4.0.3=pypi_0
attrs=23.2.0=pypi_0
bitsandbytes=0.43.1=pypi_0
blas=1.0=mkl
bzip2=1.0.8=h5eee18b_6
ca-certificates=2024.3.11=h06a4308_0
certifi=2024.2.2=pypi_0
charset-normalizer=3.3.2=pypi_0
click=8.1.7=pypi_0
contourpy=1.3.0=pypi_0
cuda-cudart=12.1.105=0
cuda-cupti=12.1.105=0
cuda-libraries=12.1.0=0
cuda-nvrtc=12.1.105=0
cuda-nvtx=12.1.105=0
cuda-opencl=12.4.127=0
cuda-runtime=12.1.0=0
cudatoolkit=11.7.0=hd8887f6_10
cycler=0.12.1=pypi_0
datasets=2.19.1=pypi_0
decorator=5.1.1=pypi_0
dill=0.3.8=pypi_0
docker-pycreds=0.4.0=pypi_0
docstring-parser=0.16=pypi_0
einops=0.8.0=pypi_0
exceptiongroup=1.2.1=pypi_0
executing=2.0.1=pypi_0
filelock=3.13.1=py310h06a4308_0
fonttools=4.53.1=pypi_0
frozenlist=1.4.1=pypi_0
fsspec=2024.3.1=pypi_0
gitdb=4.0.11=pypi_0
gitpython=3.1.43=pypi_0
gmp=6.2.1=h295c915_3
gmpy2=2.1.2=py310heeb90bb_0
huggingface-hub=0.26.2=pypi_0
idna=3.7=pypi_0
inquirerpy=0.3.4=pypi_0
intel-openmp=2023.1.0=hdb19cb5_46306
ipdb=0.13.13=pypi_0
ipython=8.26.0=pypi_0
ivon-opt=0.1.2=pypi_0
jedi=0.19.1=pypi_0
jinja2=3.1.3=py310h06a4308_0
joblib=1.4.2=pypi_0
jsonlines=4.0.0=pypi_0
kiwisolver=1.4.6=pypi_0
ld_impl_linux-64=2.38=h1181459_1
libcublas=12.1.0.26=0
libcufft=11.0.2.4=0
libcufile=1.9.1.3=0
libcurand=10.3.5.147=0
libcusolver=11.4.4.55=0
libcusparse=12.0.2.55=0
libffi=3.4.4=h6a678d5_1
libgcc-ng=11.2.0=h1234567_1
libgomp=11.2.0=h1234567_1
libnpp=12.0.2.50=0
libnvjitlink=12.1.105=0
libnvjpeg=12.1.1.14=0
libstdcxx-ng=11.2.0=h1234567_1
libuuid=1.41.5=h5eee18b_0
llvm-openmp=14.0.6=h9e868ea_0
markdown-it-py=3.0.0=pypi_0
markupsafe=2.1.3=py310h5eee18b_0
matplotlib=3.9.2=pypi_0
matplotlib-inline=0.1.7=pypi_0
mdurl=0.1.2=pypi_0
mkl=2023.1.0=h213fc3f_46344
mpc=1.1.0=h10f8cd9_1
mpfr=4.0.2=hb69a4c5_1
mpmath=1.3.0=py310h06a4308_0
multidict=6.0.5=pypi_0
multiprocess=0.70.16=pypi_0
natsort=8.4.0=pypi_0
ncurses=6.4=h6a678d5_0
networkx=3.1=py310h06a4308_0
nibabel=5.2.1=pypi_0
numpy=1.26.4=pypi_0
openssl=3.0.13=h7f8727e_1
packaging=24.0=pypi_0
pandas=2.2.2=pypi_0
parso=0.8.4=pypi_0
peft=0.10.0=pypi_0
pexpect=4.9.0=pypi_0
pfzy=0.3.4=pypi_0
pillow=10.4.0=pypi_0
pip=24.0=py310h06a4308_0
platformdirs=4.2.1=pypi_0
prompt-toolkit=3.0.47=pypi_0
protobuf=3.20.3=pypi_0
psutil=5.9.8=pypi_0
ptyprocess=0.7.0=pypi_0
pure-eval=0.2.2=pypi_0
pyarrow=16.0.0=pypi_0
pyarrow-hotfix=0.6=pypi_0
pygments=2.18.0=pypi_0
pyparsing=3.1.4=pypi_0
python=3.10.14=h955ad1f_1
python-dateutil=2.9.0.post0=pypi_0
pytorch=2.3.0=py3.10_cuda12.1_cudnn8.9.2_0
pytorch-cuda=12.1=ha16c6d3_5
pytorch-mutex=1.0=cuda
pytz=2024.1=pypi_0
pyyaml=6.0.1=py310h5eee18b_0
readline=8.2=h5eee18b_0
regex=2024.5.10=pypi_0
requests=2.31.0=pypi_0
rich=13.7.1=pypi_0
safetensors=0.4.3=pypi_0
schedulefree=1.2.6=pypi_0
scikit-learn=1.5.0=pypi_0
scipy=1.14.0=pypi_0
sentencepiece=0.2.0=pypi_0
sentry-sdk=2.1.1=pypi_0
setproctitle=1.3.3=pypi_0
setuptools=69.5.1=py310h06a4308_0
shtab=1.7.1=pypi_0
simple-parsing=0.1.5=pypi_0
six=1.16.0=pypi_0
smmap=5.0.1=pypi_0
sqlite=3.45.3=h5eee18b_0
stack-data=0.6.3=pypi_0
sympy=1.12=py310h06a4308_0
tbb=2021.8.0=hdb19cb5_0
threadpoolctl=3.5.0=pypi_0
tk=8.6.14=h39e8969_0
tokenizers=0.19.1=pypi_0
tomli=2.0.1=pypi_0
torcheval=0.0.7=pypi_0
torchtriton=2.3.0=py310
tqdm=4.66.4=pypi_0
traitlets=5.14.3=pypi_0
transformers=4.43.3=pypi_0
trl=0.8.6=pypi_0
typing_extensions=4.11.0=py310h06a4308_0
tyro=0.8.4=pypi_0
tzdata=2024.1=pypi_0
unsloth=2024.8=pypi_0
urllib3=2.2.1=pypi_0
wandb=0.17.0=pypi_0
wcwidth=0.2.13=pypi_0
wheel=0.43.0=py310h06a4308_0
xformers=0.0.26.post1=py310_cu12.1.0_pyt2.3.0
xxhash=3.4.1=pypi_0
xz=5.4.6=h5eee18b_1
yaml=0.2.5=h7b6447c_0
yarl=1.9.4=pypi_0
zlib=1.2.13=h5eee18b_1
```
