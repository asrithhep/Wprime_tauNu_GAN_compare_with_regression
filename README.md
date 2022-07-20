# Wprime_tauNu_GAN_compare_with_regression
Comparing GAN and regression model for studying the invariant mass of Wprime boson


## Signal Sample Production
The prodction details of the **W'-> tau nu** process are given below,

Sample mass| x-section <br> (pb)| no. of events 
--- | --- | --- 
3000 TeV |  0.07249 +- 4.031e-05 pb |  100000
4000 TeV  | 0.01165 +- 6.203e-06 pb  | 100000  
4500 TeV  | 0.004724 +- 2.592e-06 pb | 100000
5000 TeV |   0.00192 +- 1.078e-06 pb  | 100000

## DNN training 
The exclusion limits obtained from the test statics of **W'** invariant mass. The invariant mass obtained from the ML regression technique. The ML model can find at the ``` training/regression.py``` file. Type ```python3 training/regression.py --help``` in terminal which gives the details code inputs and usage.

```
 -h, --help            show this help message and exit
  -is INPUTSIGFILE, --inputsigFile INPUTSIGFILE
                        give the input signal root file to Test/Train
  -ib INPUTBKGFILE, --inputbkgFile INPUTBKGFILE
                        give the input bkg root file to Test/Train
  -t, --train           boolean set for train
  -s, --isSignal        boolean set for signal
  -n NEPOCH, --nepoch NEPOCH
                        no. of epochs
  -m MODEL, --model MODEL
                        model name to save or load with .h5 format

```

Once the training is performed, the model parameter saved to the ```models``` directory.

