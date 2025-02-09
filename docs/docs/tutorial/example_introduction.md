---
jupyter:
  colab:
  kernelspec:
    display_name: Python 3.10.4 ('pytorch-env')
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.10.0
  nbformat: 4
  nbformat_minor: 0
  orig_nbformat: 4
  vscode:
    interpreter:
      hash: 70d612b283d01ec4363ad8d402f22493234f9e6f823eb24b777bb7b2472b1ace
---

<div class="cell code" id="RhMRbU15NRhT">

``` python
! pip install -q torchview
! pip install -q -U graphviz
```

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:35}"
id="4Vt6oiY8NRhV" outputId="17a2d67e-9b39-4931-f86f-80939867e9e8">

``` python
from torchview import draw_graph
from torch import nn
import torch
import graphviz

# when running on VSCode run the below command
# svg format on vscode does not give desired result
graphviz.set_jupyter_format('png')
```

<div class="output execute_result" execution_count="2">

``` json
{"type":"string"}
```

</div>

</div>

<div class="cell markdown" id="1VX_jUboNRhX">

The purpose of this notebook is to introduce API and notation of
torchview package with common use cases.

</div>

<div class="cell markdown" id="ISzPg6OENRhY">

We start with simple MLP model

</div>

<div class="cell code" id="LQsuPp0ENRhY">

``` python
class MLP(nn.Module):
    """Multi Layer Perceptron with inplace option.
    Make sure inplace=true and false has the same visual graph"""

    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace),
            nn.Linear(128, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x
```

</div>

<div class="cell code" id="qyzL2xjrNRhZ">

``` python
model_graph_1 = draw_graph(
    MLP(), input_size=(2, 128),
    graph_name='MLP',
    hide_inner_tensors=False,
    hide_module_functions=False,
)
```

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:654}"
id="h7O2V3mENRhZ" outputId="a44ec58a-fe42-42ca-a980-d3e85ff35913">

``` python
model_graph_1.visual_graph
```

<div class="output execute_result" execution_count="5">

<!-- ![](a479e7729cfc48e5bba6303977af81a0bb50e63a.png) -->

</div>

</div>

<div class="cell markdown" id="5UMryxUNNRha">

Any visual graph representation of pytorch models provided by torchview
package consists of nodes and directed edges (maybe also undirected ones
for future releases). Each node is connected by an edge that indicates
information flow in the neural network.

There are 3 types of nodes:

-   Tensor Node
-   Function Node
-   Module Node

</div>

<div class="cell markdown" id="kJVDsJzCNRha">

1\) Tensor Node: This node is represented by bright yellow color. It has
the label is of the form `{tensor-name}{depth}: {tensor-shape}`.
`tensor-name` can take 3 values input-tensor, hidden-tensor, or
output-tensor. Depth is the depth of tensor in hierarchy of modules.

2\) Function Node: This node is represented by bright blue color. Its
label is of the form `{Function-name}{depth}: {input and output shape}`.

3\) Module Node: This node is represented by bright green color. Its
label is of the form `{Module-name}{depth}: {input and output shape}`.

</div>

<div class="cell markdown" id="Ae9HPM1tNRhb">

In the example of MLP above, input tensor is called by main module MLP.
This input tensor is called by its submodules, Sequential. Then, it is
called by its submodule linear. Now, inside linear module exists linear
function `F.linear`. This finally applied to input-tensor and returns
output-tensor. This is sent to ReLU layer and so on.

</div>

<div class="cell markdown" id="Uo_JGDyfNRhb">

Now, we show how rolling mechanism on recursive modules implemented. To
demonstrate this, we use RNN module

</div>

<div class="cell code" id="Sxj3gz7JNRhc">

``` python
class SimpleRNN(nn.Module):
    """Simple RNN"""

    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.hid_dim = 2
        self.input_dim = 3
        self.max_length = 4
        self.lstm = nn.LSTMCell(self.input_dim, self.hid_dim)
        self.activation = nn.LeakyReLU(inplace=inplace)
        self.projection = nn.Linear(self.hid_dim, self.input_dim)

    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        b_size = token_embedding.size()[0]
        hx = torch.randn(b_size, self.hid_dim, device=token_embedding.device)
        cx = torch.randn(b_size, self.hid_dim, device=token_embedding.device)

        for _ in range(self.max_length):
            hx, cx = self.lstm(token_embedding, (hx, cx))
            hx = self.activation(hx)

        return hx
```

</div>

<div class="cell code" id="jkPr268pNRhc">

``` python
model_graph_2 = draw_graph(
    SimpleRNN(), input_size=(2, 3),
    graph_name='RecursiveNet',
    roll=True
)
```

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:400}"
id="GRXKwRmENRhc" outputId="5a062dba-302a-439a-ed2d-1b05ce77f28e">

``` python
model_graph_2.visual_graph
```

<div class="output execute_result" execution_count="8">

<!-- ![](195da364971b52cd768860ada125ffd07f705908.png) -->

</div>

</div>

<div class="cell markdown" id="Eu1WJvTANRhd">

In the graph above, we see a rolled representation of RNN with LSTM
units. We see that LSTMCell and LeakyReLU nodes. This is representated
by the numbers show on edges. These number near edges represent the
number of edges that occur in `forward prop`. For instance, the first
number 4 represent the number of times token_embedding is used.

If the number of times that edge is used is 1, then it is not shown.

</div>

<div class="cell markdown" id="SDoq8wEhNRhd">

Another useful feature is the resize feature. Say, the previous image of
RNN is too big for your purpose. What we can do is to use resize feature
rescale it by 0.5

</div>

<div class="cell code" id="ju2uNsapNRhd">

``` python
model_graph_2.resize_graph(scale=0.5)
```

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:400}"
id="P8gQzaTGNRhd" outputId="e5c87b69-ea8a-4c0b-82c8-1f793b4f17a1">

``` python
model_graph_2.visual_graph
```

<div class="output execute_result" execution_count="10">

<!-- ![](195da364971b52cd768860ada125ffd07f705908.png) -->

</div>

</div>

<div class="cell markdown" id="5t3-aW3rNRhd">

It got smaller !!!

</div>
