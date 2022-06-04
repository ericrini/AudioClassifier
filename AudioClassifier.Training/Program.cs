using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.ML;

namespace AudioClassifier.Training;

public class Program
{
    public static async Task Main(string[] args)
    {
        IServiceProvider serviceProvider = new ServiceCollection()
            .AddLogging(logging =>
            {
                logging.SetMinimumLevel(LogLevel.Debug);
                logging.AddSimpleConsole(console => {
                    console.SingleLine = true;
                    console.TimestampFormat = "hh:mm:ss ";
                });
            })
            .AddSingleton<Trainer>()
            .AddSingleton<Preprocessor>()
            .AddSingleton<Sampler>()
            .AddSingleton<MLContext>(_ => new MLContext())
            .BuildServiceProvider();

        await serviceProvider.GetRequiredService<Trainer>()
            .TrainAsync();
    }
}