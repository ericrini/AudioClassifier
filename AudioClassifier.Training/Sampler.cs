using AudioClassifier.Training.Models;
using Microsoft.Extensions.FileSystemGlobbing;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Image;

namespace AudioClassifier.Training;

public class Sampler
{
    private readonly List<ImageData> _imageData = new();
    private readonly ILogger<Sampler> _logger;
    private readonly MLContext _mlContext;

    public Sampler(ILogger<Sampler> logger, MLContext mlContext)
    {
        _logger = logger;
        _mlContext = mlContext;
    }
    
    public IDataView? TrainingDataView { private set; get; }

    public IDataView? ValidationDataView { private set; get; }

    public IDataView? TestDataView { private set; get; }

    
    public void AddImages(string label, string directory, string glob)
    {
        _logger.LogInformation("sampler | load data set | {Label}", label);
        Matcher matcher = new();
        matcher.AddIncludePatterns(new[] {glob});

        foreach (string path in matcher.GetResultsInFullPath(directory))
        {
            _imageData.Add(new ImageData(label, path));
        }
    }

    public void PopulateDataViews(string directory)
    {
        IDataView imageDataView = _mlContext.Data.LoadFromEnumerable(_imageData);
        IDataView shuffledDataView = _mlContext.Data.ShuffleRows(imageDataView);

        IDataView processedDataView = CreateImageLoadingPipeline(directory)
            .Fit(shuffledDataView)
            .Transform(shuffledDataView);

        DataOperationsCatalog.TrainTestData trainSplit = _mlContext.Data.TrainTestSplit(processedDataView, 0.3);
        DataOperationsCatalog.TrainTestData validationSplit = _mlContext.Data.TrainTestSplit(trainSplit.TestSet, 0.1);

        TrainingDataView = trainSplit.TrainSet;
        ValidationDataView = validationSplit.TrainSet;
        TestDataView = validationSplit.TestSet;
    }

    private EstimatorChain<ImageLoadingTransformer> CreateImageLoadingPipeline(string directory)
    {
        ValueToKeyMappingEstimator valueToKeyMappingEstimator = _mlContext.Transforms.Conversion.MapValueToKey
        (
            nameof(ModelInput.LabelAsKey), 
            nameof(ModelInput.Label)
        );
        
        ImageLoadingEstimator imageLoadingEstimator = _mlContext.Transforms.LoadRawImageBytes
        (
            inputColumnName: nameof(ModelInput.ImagePath), 
            outputColumnName: nameof(ModelInput.Image), 
            imageFolder: directory
        );
        
        return valueToKeyMappingEstimator.Append(imageLoadingEstimator).Append(imageLoadingEstimator);
    }
}