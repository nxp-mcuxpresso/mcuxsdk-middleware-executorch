# Overview

ExecuTorch is an end-to-end solution for enabling on-device inference capabilities across mobile and edge devices including wearables, embedded devices and microcontrollers. It is part of the PyTorch Edge ecosystem and enables efficient deployment of PyTorch models to edge devices. For more information, see https://pytorch.org/executorch-overview.

The MCUXpresso Software Development Kit \(MCUXpresso SDK\) provides a comprehensive software package with a pre-integrated ExecuTorch based on version v1.2.0 which includes the Neutron Backend. The Neutron Backend enables acceleration of ML models on the [eIQ® Neutron Neural Processing Unit (NPU)](https://www.nxp.com/applications/technologies/ai-and-machine-learning/eiq-neutron-npu:EIQ-NEUTRON-NPU).

This document describes the steps required to download and start using ExecuTorch. Additionally, the document describes the steps required to create an application for running pre-trained models.

**Note:** The document also assumes knowledge of machine learning frameworks for model training.

## Supported platforms: 
* [i.MX RT700](https://www.nxp.com/products/i.MX-RT700)

## Installation
To use ExecuTorch with the Neutron Backend, you will need:
* ExecuTorch from https://github.com/nxp-mcuxpresso/mcuxsdk-middleware-executorch.git
* eIQ Neutron SDK
* MCUXpresso SDK

Here we briefly describe each component's purpose and steps to install them.

The **ExecuTorch AoT** and **eIQ Neutron SDK** are needed to convert a PyTorch model to ExecuTorch and delegate it to eIQ Neutron NPU using the Neutron Backend. 
The **MCUXpresso SDK** provides project to build the ExecuTorch Runtime Library, an example application with a simple CNN, toolchains and other middleware libraries to build and deploy the application on the target platform.

If you want to run the prepared example application on the i.MX RT700 platform, and skip the model preparation phase, continue with the [MCUXpresso SDK Part](#mcuxpresso-sdk-part). 

### ExecuTorch for Ahead of Time model preparation
The ExecuTorch enables to deploy PyTorch models on edge devices. For this purpose the PyTorch model must be processed and converted by the ExecuTorch Ahead of Time (AoT) part. You can obtain the full ExecuTorch including the AoT part aligned with this version of the MCUXpresso SDK from the [mcuxsdk-middleware-executorch](https://github.com/nxp-mcuxpresso/mcuxsdk-middleware-executorch/tree/release/mcux-full) release/mcux-full branch.

#### Installation
Prerequisites:
* x86 Linux machine with GLIBC-2.29 or higher (e.g. Ubuntu 20.04 or higher)
* Python 3.10, 3.11, 3.12 or 3.13

To build and install the ExecuTorch follow these steps:

1. (Optional) Setup Python virtual environment on desired location and activate it.
```commandline
$ python3 -m venv venv
$ source venv/bin/activate
``` 

2. Clone ExecuTorch from [mcuxsdk-middleware-executorch](https://github.com/nxp-mcuxpresso/mcuxsdk-middleware-executorch/tree/release/mcux-full)
```commandline
$ git clone --branch release/mcux-full https://github.com/nxp-mcuxpresso/mcuxsdk-middleware-executorch.git
$ cd mcuxsdk-middleware-executorch
$ git submodule update --init --recursive
```

3. Build and install ExecuTorch and its dependencies:
```commandline
$ ./install_executorch.sh
```
> [!WARNING]
> The `install_requirements.sh` installs the CPU version of `torch` from `https://download.pytorch.org/whl/cpu`. If you are behind corporate proxy, it might have issues accessing it and you will see warnings like: 
> ```commandline
>  WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1006)'))': /whl/test/cpu/torch/
>  ```
> In this case the CUDA version of torch is installed and the `install_requirements.sh` script fails with: 
> ```commandline
> PyTorch: CUDA cannot be found.  Depending on whether you are building
> ```
> Make sure pip can access the `https://download.pytorch.org/whl/cpu` PyPI.

Next continue with the installation of the [Neutron Converter](#neutron-converter)

### Neutron Converter
The eIQ Neutron Backend uses the Neutron Converter from eIQ Neutron SDK to convert the ExecuTorch program to the eIQ Neutron NPU microcode.

#### Installation
The Neutron Converter and the Neutron Simulator are available as Python packages and can be installed by the `pip` command from eiq.nxp.com/repository:
```commandline
pip install --index-url https://eiq.nxp.com/repository eiq-neutron-sdk==3.1.1 eiq_nsys
```
Or you can use the prepared setup script: 
```commandline
./examples/nxp/setup.sh
```

### MCUXpresso SDK
The MCUXpresso SDK is used to build, debug and deploy the application using ExecuTorch on the target platform.

You can obtain the MCUXpresso SDK from [MCUXpresso SDK Builder](https://mcuxpresso.nxp.com/en) including the IDE. 
See the [getting_mcuxpresso](getting_mcuxpresso.md) for details. 

In the MCUXpresso SDK, there are 2 projects available related to ExecuTorch: 
* executorch_lib
* executorch_cifarnet

For more details see [example_applications](example_applications.md). Here you will find the details on how to build and run the demo applications.
