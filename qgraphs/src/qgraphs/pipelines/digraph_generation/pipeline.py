from kedro.pipeline import Pipeline, node, pipeline

from .nodes import graph_generator, convert_to_adj, digraph_adj_expand, digraph_adj_weigh, graph_gen
def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=graph_generator,
                inputs="params:new_graph",
                outputs="digraph_unw",
                name="create_digraph",
            ),
            node(
                func=convert_to_adj,
                inputs="digraph_unw",
                outputs="adj_graph",
                name="create_adj",
            ),
            node(
                func=digraph_adj_expand,
                inputs="adj_graph",
                outputs="adj_exp",
                name="create_adj_exp",
            ),
            node(
                func=digraph_adj_weigh,
                inputs=["adj_exp","params:digraph_adj_weight"],
                outputs="adj_w",
                name="create_adj_weigh",
            ),
            node(
                func=graph_gen,
                inputs="adj_w",
                outputs="digraph_wei",
                name="create_digraph_weighted",
            ),
        ],
    )
