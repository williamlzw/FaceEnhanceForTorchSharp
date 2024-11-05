using Microsoft.ML.OnnxRuntime;
using TorchSharp;
using static FaceEnhance.Utils;

namespace FaceEnhance
{
    public class FaceEnhance_Codeformer
    {
        private readonly InferenceSession _session;

        public FaceEnhance_Codeformer(string modelFile)
        {
            SessionOptions options = new SessionOptions();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
            // 设置为CPU上运行
            options.AppendExecutionProvider_CPU(0);
            _session = new InferenceSession(modelFile, options);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="img"></param>
        /// <param name="facelandmark5"></param>
        /// <param name="threshold">0.1~1.0</param>
        /// <returns></returns>
        public torch.Tensor Enhance(torch.Tensor img, torch.Tensor facelandmark5, double threshold = 1.0)
        {
            img = img.unsqueeze(0).to(torch.float32);
            var (warpSrc, affineMatrix) = WarpFaceByFaceLandmark5(img, facelandmark5, WarpType.FFHQ_512, new torch.Size(new int[] { 512, 512 }));
            var inputTensor = Preprocess(warpSrc);
            var runOptions = new RunOptions();
            var ortValue1 = OrtValue.CreateTensorValueFromMemory(inputTensor.data<float>().ToArray(), new long[] { 1, 3, 512, 512 });
            var ortValue2 = OrtValue.CreateTensorValueFromMemory(new double[] { threshold }, new long[] { 1 });
            var inputs = new Dictionary<string, OrtValue>
             {
                 { "input", ortValue1 },
                 { "weight", ortValue2 }
             };
            var results = _session.Run(runOptions, inputs, _session.OutputNames);
            var enhanceResult = results[0].GetTensorDataAsSpan<float>().ToArray();
            var enhanceTensor = torch.from_array(enhanceResult).reshape(1, 3, 512, 512);
            var normalizedTensor = Postprocess(enhanceTensor);
            var cropMask = CreateBoxMask(normalizedTensor.shape[2], normalizedTensor.shape[3]);
            var pasteTensor = PasteBack(img, normalizedTensor, cropMask, affineMatrix.unsqueeze(0));
            var blendTensor = BlendImg(img, pasteTensor);
            return pasteTensor;
        }
    }
}
