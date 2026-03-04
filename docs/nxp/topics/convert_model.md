# PyTorch Model Conversion to ExecuTorch Format

In this guideline we will show how to use the ExecuTorch AoT part to convert a PyTorch model to ExecuTorch format and delegate the model computation to eIQ Neutron NPU using the eIQ Neutron Backend.

First we will start with an example script converting the model. This example shows the CifarNet model preparation. It is the same model which is part of the `example_cifarnet`.

1. Run the `aot_neutron_compile.py` example with the `cifar10` model. 
As the `aot_neutron_compile.py` is already installed as part of the ExecuTorch installation, we will run it from there 
```commandline
$ python -m examples.nxp.aot_neutron_compile --quantize \
        --delegate --remove-quant-io-ops --use_channels_last_dim_order \
        -m cifar10
```

2. This command generates a `cifar10_nxp_delegate.pte` file, which can be used with the MCUXpresso SDK `cifarnet_example` project. 

The generated PTE file is used in the executorch_cifarnet example application, see [example_applications](example_applications.md).

## NXP eIQ Neutron Kernel Selective Kernel Registration

The NXP ExecuTorch backend supports selective Neutron kernel registration for Neutron-C targets, which reduces the size of the Neutron Firmware.
For more information and example commands, see [NXP eIQ Neutron Kernel Selective Kernel Registration](../../source/backends/nxp/nxp-kernel-selection.md).

### MCUXpresso SDK build with kernel selection

The build process of only selected kernels to reduce the size of deployed application is described in
[MCUXpresso SDK build with kernel selection](example_applications.md#mcuxpresso-sdk-build-with-kernel-selection)