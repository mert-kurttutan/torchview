# `__torch_function__`

本文档解释 `__torch_function__` 在本项目中的作用，以及若干与版本相关的注意事项。

- `__torch_function__` API 最早在 v1.5.0 引入，见：[PyTorch v1.5.0 Release Notes](https://github.com/pytorch/pytorch/releases/tag/v1.5.0)

- 一些重要算子对 subclass 的保留（subclass preservation）在 v1.7.0 引入，见：[PyTorch v1.7.0 Release Notes](https://github.com/pytorch/pytorch/releases/tag/v1.7.0)。这一点非常关键，因为 `torchview` 只追踪 `RecorderTensor`（`torch.Tensor` 的子类）产生的 tensor 流。

- v1.9.0 引入了一些重要修复，例如对 `F.embedding` 的支持，见：[PyTorch v1.9.0 Release Notes](https://github.com/pytorch/pytorch/releases/tag/v1.9.0)。在更早版本中，subclass 的 `__torch_function__` 下调用 `F.embedding` 可能返回 `NotImplemented`，导致返回值退化为普通 `torch.Tensor`（而不是 `RecorderTensor`），这不是我们期望的行为。

为了避免该问题（并兼容 PyTorch < 1.9），我们在 `recorder_tensor.py` 中加入了如下逻辑：

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

更精确地说：

```
F.linear 在 1.7.1、1.8、1.9 会返回 `NotImplemented`
F.embedding 在 1.7.1、1.8、1.9 会返回 `NotImplemented`
```

此外，本包不支持 torch 版本 \(\le 1.6\)，因为这些版本的 `torch.Tensor` 还不支持以类方法形式提供 `__torch_function__`。

- 其它相关 PR / issue：
  - [PR1](https://github.com/pytorch/pytorch/pull/32799)
  - [PR2](https://github.com/pytorch/pytorch/issues/24015)

## Meta tensor 相关链接

- https://github.com/pytorch/pytorch/blob/orig/release/1.9/torch/_tensor.py
- https://github.com/pytorch/pytorch/issues/87990


