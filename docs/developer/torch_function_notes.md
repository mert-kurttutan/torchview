# __torch_function__

This document explains some important point about `__torch_function__` and its role in this project.

* `__torch_function__` API is first introduced in v1.5.0, [here](https://github.com/pytorch/pytorch/releases/tag/v1.5.0)

* Subclass preservations of some important operators introduced [here](https://github.com/pytorch/pytorch/releases/tag/v1.7.0). This property is very important since `torchview` package keeps track tensor of only `RecorderTensor` subclass. 


* Some important fixes introdued [here](https://github.com/pytorch/pytorch/releases/tag/v1.9.0). For instance support for `F.embedding` is included. Otherwise, `F.embedding` under `__torch_function__` of subclass would return `NotImplemented`, leading to `torch.Tensor`, which is not desired. To prevent this issue (and support pytorch version < 1.9), we added the below code in `recorder_tensor.py`


```
        # This is necessary for torch version < 1.10
        if func in [F.linear, F.embedding]:
            out = nn.parameter.Parameter.__torch_function__(
                func, types, args, kwargs).as_subclass(RecorderTensor)
        else:
            # use original torch_function; otherwise,
            # it leads to infinite recursive call of torch_function
            out = super().__torch_function__(func, types, args, kwargs)
```
To be precise about the versions,
```
F.linear returns `NotImplemented` for versions 1.7.1, 1.8, 1.9 
F.embedding returns `NotImplemented` for versions 1.7.1, 1.8, 1.9 
```

This package does not support torch version <= 1.6 since torch.Tensor does not have `__torch_function__` as class methods in these version.

* Some other relevant PRs: [PR1](https://github.com/pytorch/pytorch/pull/32799), [PR2](https://github.com/pytorch/pytorch/issues/24015)


# Meta tensor related links
* https://github.com/pytorch/pytorch/blob/orig/release/1.9/torch/_tensor.py
* https://github.com/pytorch/pytorch/issues/87990
