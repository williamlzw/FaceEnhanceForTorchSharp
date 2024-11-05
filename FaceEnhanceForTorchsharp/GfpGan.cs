using Microsoft.ML.OnnxRuntime;
using TorchSharp;
using static FaceEnhance.Utils;

namespace FaceEnhance
{
    public class FaceEnhance_GfpGan
    {
        private readonly InferenceSession _session;

        public FaceEnhance_GfpGan(string modelFile)
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
        /// <returns></returns>
        public torch.Tensor Enhance(torch.Tensor img, torch.Tensor facelandmark5)
        {
            img = img.unsqueeze(0).to(torch.float32);
            var (warpSrc, affineMatrix) = WarpFaceByFaceLandmark5(img, facelandmark5, WarpType.FFHQ_512, new torch.Size(new int[] { 512, 512 }));
            var inputTensor = Preprocess(warpSrc);
            var runOptions = new RunOptions();
            var ortValue = OrtValue.CreateTensorValueFromMemory(inputTensor.data<float>().ToArray(), new long[] { 1, 3, 512, 512 });
            var inputs = new Dictionary<string, OrtValue>
             {
                 { "input", ortValue }
             };
            var results = _session.Run(runOptions, inputs, _session.OutputNames);
            var enhanceResult = results[0].GetTensorDataAsSpan<float>().ToArray();
            var enhanceTensor = torch.from_array(enhanceResult).reshape(1, 3, 512, 512);
            var normalizedTensor = Postprocess(enhanceTensor);
            var cropMask = CreateBoxMask(normalizedTensor.shape[2], normalizedTensor.shape[3]);
            var pasteTensor = PasteBack(img, normalizedTensor, cropMask, affineMatrix.unsqueeze(0));
            var blendTensor = BlendImg(img, pasteTensor);
            return blendTensor;
        }
    }
}
