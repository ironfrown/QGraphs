"""Project pipelines."""

from kedro.pipeline import Pipeline

from qgraphs.pipelines import digraph_generation as gen
# from qgraphs.pipelines import model_analysis as ma
# from qgraphs.pipelines import model_training as mt

def register_pipelines() -> dict[str, Pipeline]:
    # model_analysis_pipeline = ma.create_pipeline()
    # model_training_pipeline = mt.create_pipeline()
    digraph_generation_pipeline = gen.create_pipeline()

    return {
        # "ma": model_analysis_pipeline,
        # "mt": model_training_pipeline,
        # "__default__": model_training_pipeline + model_analysis_pipeline
        "__default__": digraph_generation_pipeline
    }
