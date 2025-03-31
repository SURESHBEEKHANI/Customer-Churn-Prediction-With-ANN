from src.pipelines.training_pipeline import Pipeline

if __name__ == "__main__":
    pipeline = Pipeline()
    trained_model = pipeline.run_pipeline()
    print("Trained model:", trained_model)
