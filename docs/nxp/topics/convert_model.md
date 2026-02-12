# PyTorch Model Conversion to ExecuTorch Format

In this guideline we will show how to use the ExecuTorch AoT part to convert a PyTorch model to ExecuTorch format and delegate the model computation to eIQ Neutron NPU using the eIQ Neutron Backend.

First we will start with an example script converting the model. This example shows the CifarNet model preparation. It is the same model which is part of the `example_cifarnet`.

1. Run the `aot_neutron_compile.py` example with the `cifar10` model. 
As the `aot_neutron_compile.py` is already installed as part of the ExecuTorch installation we will run it from there 
```commandline
$ python -m examples.nxp.aot_neutron_compile --quantize \
        --delegate --neutron_converter_flavor SDK_26_03 -m cifar10
```

2. It will generate you `cifar10_nxp_delegate.pte` file which can be used with the MXUXpresso SDK `cifarnet_example` project. 

The generated PTE file is used in the executorch_cifarnet example application, see [example_application](example_applications.md).
 
