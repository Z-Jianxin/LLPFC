# LLPFC
1. Dependencies:
   * download and install Anaconda
   * create an environment and install the dependencies: `conda env create -f LLPFC.yml`
   * activate the new environment: `conda activate LLP`
2. Generate LLP data:
   * run `python make_data.py -h` to check the usage
   * Example: `python make_data.py -d cifar10 -c 10 -s cifar10_paper_BS256_BN160_T0 \
                    -l ./data/labeled_data/  \
                    -p ./data/llp_data/ \
                    -m dirichlet -b 256 -n 160 -r 15671628`
   * The flag `-m` allows options `dirichlet` and `uniform`. Experiments in the paper use `-m dirichlet`.
3. Run Experiments:
   * run `python main.py -h` to check the usage
   * Example: `python main.py -d cifar10 -c 10 -v 0 -a llpfc --seed 0 \  
                -p ./data/llp_data/cifar10_paper_BS256_BN160_T0 \  
                -f ./data/labeled_data/ \  
                -n wide_resnet_d_w -wrnd 16 -wrnw 4 -e 200 -r 20 -b 128 -dr 0.3 \  
                -o nesterov -l 1e-1 -w uniform -wd 5e-4 -m 0.9 -sc drop -ms 60 120 160 -ga 0.2 \  
                -log ./log_llpfc_BS256_BN160_T0 \  
                -s ./llpfc_BS256_BN160_T0.pth`