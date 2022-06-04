namespace AudioClassifier.Training.Models;

public class ImageData
{
    public ImageData(string label, string path)
    {
        Label = label;
        ImagePath = path;
    }
    
    public string ImagePath { get; }

    public string Label { get; }
}