# OpenHD 
OpenHD - A GPU-Powered Framework for Hyperdimensional Computing
* Author: Jaeyoung Kang (UCSD) and Yeseong Kim (DGIST)

The OpenHD framework enables the GPU-based execution of HD Computing using JIT-like compliation written in Python fo high efficiency.
For the implementation details, please refer to our papers in the references section below.

## Requirements
We included the library dependencies in the pip installer.
You also need to install GraphViz for debuging purpose.
```bash
# apt install graphviz
```

## Install
You can install the OpenHD framework using pip3:
```bash
pip install .
```

## Usage
An usage example is provided in example/voicehd.py:
```bash
python3 examples/voicehd.py -t examples/dataset/isolet_train.choir_dat -i examples/dataset/isolet_test.choir_dat
```

## References
```
@article{kang2022openhd,
  title={OpenHD: A GPU-Powered Framework for Hyperdimensional Computing},
  author={Kang, Jaeyoung and Khaleghi, Behnam and Rosing, Tajana and Kim, Yeseong},
  journal={IEEE Transactions on Computers},
  year={2022},
  publisher={IEEE}
}

@inproceedings{kang2022xcelhd,
  title={XCelHD: An efficient GPU-powered hyperdimensional computing with parallelized training},
  author={Kang, Jaeyoung and Khaleghi, Behnam and Kim, Yeseong and Rosing, Tajana},
  booktitle={2022 27th Asia and South Pacific Design Automation Conference (ASP-DAC)},
  pages={220--225},
  year={2022},
  organization={IEEE}
}
```
