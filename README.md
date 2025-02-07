## Quantum QGraph in Networkx+PennyLane+PyTorch

### Aims
*To create quantum graphs capable of using quantum mechanical principles of
graph representation, operation and use.*

### Current work v1.18+
*Application of qgraphs to determine vertext-to-vertex reachability in n-steps*
- Alter utils.Models to allow creation of n-steps models after training.<br/>
  This would allow training a 1-step (adjacency) model and then using such models<br/>
  (and their trained params) to analyse node-to-node
  reachability in n-steps (with some probability).
- Subsequently, create a notebook `pl_qgraphs_app_v1_XX_n_steps.ipynb`, to test n_steps extension.
- Add explanation of changes in v1.18+ in  `pl_qgraphs_main_v1_XX_versions.ipynb`.
- Add mathematical foundations for n_step analysis in `pl_qgraphs_main_v1_XX_versions.ipynb`.

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
