using TorchSharp;
using FaceDetect;

namespace FaceEnhance
{
    public class Test
    {
        public static void Main()
        {
            var toolDetect = new FaceDetect_Yolo8nface("yoloface_8n.onnx");
            var imgPath = "data\\test.jpg";
            torchvision.io.DefaultImager = new torchvision.io.SkiaImager();
            var imgTensor = torchvision.io.read_image(imgPath);

            var time0 = DateTime.Now;
            var (box, facelandmark5, score) = toolDetect.Detect(imgTensor);
            var time1 = DateTime.Now;
            Console.WriteLine("detect time:" + (time1 - time0).TotalMilliseconds);

            //var toolFaceEnhance = new FaceEnhance_GfpGan("gfpgan_1.4.onnx");
            var toolFaceEnhance = new FaceEnhance_Codeformer("codeformer.onnx");
            var time2 = DateTime.Now;
            var resultTensor = toolFaceEnhance.Enhance(imgTensor, facelandmark5, 0.4);
            var time3 = DateTime.Now;
            Console.WriteLine("enhance time:" + (time3 - time2).TotalMilliseconds);
            torchvision.io.write_jpeg(resultTensor.to(torch.uint8), "result.jpg");
        }
    }
}