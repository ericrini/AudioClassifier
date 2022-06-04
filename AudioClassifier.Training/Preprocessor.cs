using FFMpegCore;
using Microsoft.Extensions.FileSystemGlobbing;
using Microsoft.Extensions.Logging;
using NAudio.Wave;
using Spectrogram;

namespace AudioClassifier.Training;

public class Preprocessor
{
    private readonly ILogger<Preprocessor> _logger;

    public Preprocessor(ILogger<Preprocessor> logger)
    {
        _logger = logger;
    }

    public async Task CreateSpectrogramsAsync(string directory, string glob, int sampleRate = 8000, int width = 1000, int parallelism = 1)
    {
        SemaphoreSlim semaphoreSlim = new SemaphoreSlim(parallelism, parallelism);
        Matcher matcher = new();
        matcher.AddIncludePatterns(new[] {glob});

        foreach (string path in matcher.GetResultsInFullPath(directory))
        {
            await semaphoreSlim.WaitAsync();
            _logger.LogDebug("preprocessor | create spectrogram | {File} in {Directory}", Path.GetFileName(path), Path.GetDirectoryName(path));
            
            _ = CreateSpectrogramAsync(path, sampleRate, width)
                .ContinueWith(_ => semaphoreSlim.Release());
        }
    }

    private async Task CreateSpectrogramAsync(string path, int sampleRate, int width)
    {
        try
        {
            string waveOutputPath = path.Replace("m4a", "wav");
            string spectrogramOutputPath = path.Replace("m4a", "bmp");

            if (File.Exists(waveOutputPath))
            {
                File.Delete(waveOutputPath);
            }
        
            if (File.Exists(spectrogramOutputPath))
            {
                File.Delete(spectrogramOutputPath);
            }
            
            await FFMpegArguments
                .FromFileInput(path)
                .OutputToFile(waveOutputPath, true, options =>
                {
                    options.WithAudioSamplingRate(sampleRate);
                })
                .ProcessAsynchronously();

            SpectrogramGenerator spectrogram;
            
            await using (WaveStream waveStream = new WaveFileReader(waveOutputPath))
            {
                long stepSize = waveStream.Length / waveStream.BlockAlign / width;
                spectrogram = new(sampleRate, fftSize: 2048, stepSize: (int) stepSize, maxFreq: 3000);
                float[] samples = new float[1024];
                
                ISampleProvider sampleProvider = waveStream.ToSampleProvider();
                int samplesRead = sampleProvider.Read(samples, 0, samples.Length);
            
                while (samplesRead > 0)
                {
                    spectrogram.Add(samples.Select(sample => (double) sample * 16000));
                    samplesRead = sampleProvider.Read(samples, 0, samples.Length);
                }
            }

            spectrogram.GetBitmapMel(dB: true, intensity: 6).Save(spectrogramOutputPath);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "preprocessor | spectrogram creation failed | {File} in {Directory}", Path.GetFileName(path), Path.GetDirectoryName(path));
        }
    }
}