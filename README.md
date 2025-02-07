## Quantum QGraph in Networkx+PennyLane+PyTorch

### Folders
- legacy: previous versions dated with their time of their removal (includes legacy utilities)
- logs: log and database of qgraph development tasks, description below
- runs: important runs of various notebooks (some of historical value only)
- utils: which is a collection of Python files and testing notebooks to manage QGraphs, plots and quantum circuits
- versions: previous versions of notebooks
  
### Important notebooks
- pl_qgraphs_main: the main notebook with documentation and version lists
- pl_qgraphs_data: creates sample digraphs
- pl_qgraphs_train: trains a qgraph model
- pl_qgraphs_analz: analyses a qgraph model
- pl_qgraphs_app: application of a qgraph model

### Log folder structure
Log files related to graph representation have the following naming convention:
- pattern: `digraph_{vertex #}_{edge #}_{unw | wei}.json`<br/>
- example: `digraph_064_039_v010_unw.json`

Log files and folders related to training have the following naming convention:
- pattern:<br/>
  `case_data_v{vertex #}_e{edge #}_n{data qubits #}_x{extra qubits #}...`<br/>
  `..._lays{layers #}_{rotation gate type}_{mode}_ep{epoch #}_{other}.{eps | json}`<br/>
- example: `bench_rand_v8_e14_n3_x1_lays3_Rxyz_cs8_ep2000_graph.eps`

Logs are located in the following sub-folders:
- data: Additional data files
- graphs: definition of unweighted (unw) and weighted (wei) digraphs (in networkx format), naming convention:
- figures: figures and diagrams in 'eps' format
- training: training hyper-parameters, model parameters, input and output graphs
- analysis: analysis of model runs and their results
- legacy_graphs: legacy folder "sample_graphs" with some of the older test graphs included
...
