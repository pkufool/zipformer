Just a placeholder, will update full package soon.

### reproduce

baseline
--epoch 50 --avg 24: 2.12 / 4.62

exp-libri-reproduce
epoch 150 avg 30:  2.14 / 4.89

exp-libri-reproduce2
epoch 50 avg 9: 2.07 / 4.86

exp-libri-reproduce3
epoch 50 avg 22: 2.12 / 4.66

exp-libri-reproduce4
epoch 50 avg 19: 2.05 / 4.75


#### large

baseline
epoch 50 avg 26: 1.9 / 3.96

exp-large-cr-ctc-rnnt
epoch 150 avg 30: 1.92 / 4.15

exp-large-cr-ctc-rnnt2
epoch 50 avg :



pip install pre-commit
cd zipformer
pre-commit install
