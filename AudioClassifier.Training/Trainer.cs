using AudioClassifier.Training.Models;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Vision;

namespace AudioClassifier.Training;

public class Trainer
{
    private readonly ILogger<Trainer> _logger;
    private readonly Preprocessor _preprocessor;
    private readonly Sampler _sampler;
    private readonly MLContext _mlContext;

    public Trainer(ILogger<Trainer> logger, Preprocessor preprocessor, Sampler sampler, MLContext mlContext)
    {
        _logger = logger;
        _preprocessor = preprocessor;
        _sampler = sampler;
        _mlContext = mlContext;
    }

    public async Task TrainAsync()
    {
        try
        {
            await Task.Delay(0);
            //await _preprocessor.CreateSpectrogramsAsync("../../../Data", "**/*.m4a", 8000, 1000, 8);
            
            _sampler.AddImages("male", "../../../Data/males", "*.bmp");
            _sampler.AddImages("female", "../../../Data/females", "*.bmp");
            _sampler.PopulateDataViews("../../../Data");

            ImageClassificationTrainer.Options options = new ImageClassificationTrainer.Options
            {
                FeatureColumnName = nameof(ModelInput.Image),
                LabelColumnName = nameof(ModelInput.LabelAsKey),
                ValidationSet = _sampler.ValidationDataView,
                Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
                MetricsCallback = (metrics) => _logger.LogInformation("ml.net | {Metrics}", metrics),
                TestOnTrainSet = false,
                ReuseTrainSetBottleneckCachedValues = true,
                ReuseValidationSetBottleneckCachedValues = true
            };
            
            ImageClassificationTrainer trainer = _mlContext.MulticlassClassification.Trainers.ImageClassification(options);
            KeyToValueMappingEstimator estimator = _mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel");
            EstimatorChain<KeyToValueMappingTransformer> pipeline = trainer.Append(estimator);
            pipeline.Fit(_sampler.TrainingDataView);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "unhandled exception from training pipeline");
        }
    }
}