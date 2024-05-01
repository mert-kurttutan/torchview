# 30/4/24 DH:
"""
Written to add a legend to the generated GraphViz graph from 'torchview'

Necessary alterations to 'torchview' codebase:
----------------------------------------------
SUMMARY:
--------
1) Add API call to 'draw_graph()'
2) Store Torch component details in ModuleNode
3) Add Dictionary to ComputationGraph to collate data
4) Populate ComputationGraph Dictionary
5) Add legend key to GraphViz component

DETAILS:
--------
1) Add API call to 'draw_graph()'.
torchview.py::draw_graph(…)
  ...
  model_graph.fill_visual_graph()

  # Legend API call
  addLegendToGV(model_graph, filename)
  ...

2) Add member variable to ModuleNode to hold the Torch component details 
   (prior to populating 'ComputationGraph.module_unit_dict').
compute_node.py::ModuleNode(Node).__init__(…)
  ...
  # 
  self.module_unit_ID = module_unit
  ...

3) Add Dictionary to ComputationGraph to collate data to be added to the graph.
computation_graph.py::ComputationGraph.reset_graph_history()
  ...
  # Dict to store 'module_unit' for later compare with model print out
  self.module_unit_dict = {}
  ...

4) Add code to hold the Torch component details in ComputationGraph.module_unit_dict' via component number key.
computation_graph.py::ComputationGraph.add_node(…)
  ...
  # Necessary to add legend to node graph
  if hasattr(node, 'module_unit_ID'):
    self.module_unit_dict[self.id_dict[node.node_id]] = node.module_unit_ID

5) Alter the GraphViz HTML generating function to add the component number (ie the legend key)
computation_graph.py::ComputationGraph.get_node_label(…)

  if isinstance(node, TensorNode):
    ...
  else:
    input_repr = compact_list_repr(node.input_shape)
    output_repr = compact_list_repr(node.output_shape)

    # ----------------------------------------------------------------------------------------------------
    module_unit_num = ""
    if hasattr(node, 'module_unit_ID'):

      # Adding legend # to node box
      module_unit_num = f"#{self.id_dict[node.node_id]}"
    # ----------------------------------------------------------------------------------------------------

    # Added '{module_unit_num}' to '{node.name}' row
    label = f'''<
        <TABLE BORDER="{border}" CELLBORDER="{cell_bor}"
        CELLSPACING="{cell_sp}" CELLPADDING="{cell_pad}">
        <TR>
          <TD ROWSPAN="2">{module_unit}<BR/>{node.name}<BR/>depth:{node.depth}</TD>


"""
import graphviz
# -----------------------------------------------   API  ----------------------------------------------------
def addLegendToGV(model_graph, filename):
    # 29/4/24 DH: "Node #"       -> "descStr hash" | "Node #"
    #             "descStr hash" -> "descStr"
    nodeNumDict = {}
    descHashDict = {}
    for nodeElem in model_graph.module_unit_dict.keys():
        descStr = str(model_graph.module_unit_dict[nodeElem])
        hashVal = hash(descStr)

        if hashVal in descHashDict:
            # When the 'descStr' has already been used (ie repeating Attention) then we need to store the matching node #
            # Find matching node # by finding key from value:
            #   https://www.geeksforgeeks.org/python-get-key-from-value-in-dictionary/
            matchingNodeNum = list(nodeNumDict.keys())[list(nodeNumDict.values()).index(hashVal)]
            nodeNumDict[nodeElem] = matchingNodeNum
        else:
            nodeNumDict[nodeElem] = hashVal
            descHashDict[hashVal] = descStr

    # 29/4/24 DH: Need to call 'visual_graph.render()' in order to save .gv file for legend addition
    model_graph.visual_graph.render()
    _addLegendToGV(nodeNumDict, descHashDict, filename)
    

# ----------------------------------------------- INTERNAL --------------------------------------------------
# 29/4/24 DH:
def printNodeDicts(nodeNumDict, descHashDict):
    print()
    print("Collated node #'s with description hash OR matching node #")
    print("----------------------------------------------------------")
    for elem in nodeNumDict.keys():
        print(f"{elem}: {nodeNumDict[elem]}")
    print("----------------------------------------------------------")
    print()
    print("Description hash dictionary")
    print("---------------------------")
    for elem in descHashDict.keys():
        print(f"{elem}: {descHashDict[elem]}")
    print("---------------------------")
    print()

# 29/4/24 DH:
def addLegendLabel(data2DList):
    dataLines = []
    for pair in data2DList:
      nodeNum = pair[0]
      # 29/4/24 DH: Make single item lists look better by removing the "[]" in the <TD>
      if len(nodeNum) == 1:
          nodeNum = nodeNum[0]
      # Newlines need to be replaced by "<BR/>" in Component str
      comp = pair[1]
      if "\n" in comp:
          comp = comp.replace("\n", "<BR/>")

      line = f"<TR><TD>{nodeNum}</TD><TD>{comp}</TD></TR>"
      dataLines.append(line)

    label = f'''  legend [label=<
                    <TABLE BORDER="0" CELLBORDER="1"
                    CELLSPACING="0" CELLPADDING="4">
                        {dataLines}
                    </TABLE>> fillcolor=darkseagreen1]'''
    return label

# 29/4/24 DH:
def addLegendBlock(nodeNumDict, descHashDict, fp):
    
    data2DList = []
    for key in nodeNumDict.keys():
        try:
            pair = []
            descStr = descHashDict[nodeNumDict[key]]
            pair.append([key])
            pair.append(descStr)
            data2DList.append(pair)
        except KeyError as e:
            prevComp = nodeNumDict[key]
            for pair in data2DList:
                if prevComp in pair[0]:
                    pair[0].append(key)
    
    label = addLegendLabel(data2DList)
    fp.write(label + "\n")
    
            
# 29/4/24 DH:
def _addLegendToGV(nodeNumDict, descHashDict, filename):
    printNodeDicts(nodeNumDict, descHashDict)

    textLines = []
    with open(filename, 'r') as fp:
        textLines = [line for line in fp.readlines()]
        
    # Remove the trailing "}"
    textLines.pop()
    
    with open(filename, 'w') as fp:
        for line in textLines:
            fp.write(line)
        
        addLegendBlock(nodeNumDict, descHashDict, fp)
        # Readd the removed closing brace after adding the legend
        fp.write("}")

    import pathlib
    fp = pathlib.Path(filename)
    graphviz.render('dot', 'pdf', fp).replace('\\', '/')
    
    print()

