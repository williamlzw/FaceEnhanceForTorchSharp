using TorchSharp;
using static TorchSharp.torch;

namespace FaceEnhance
{
    public class Utils
    {
        public enum WarpType
        {
            Arcface_112_v1,
            Arcface_112_v2,
            Arcface_128_v2,
            FFHQ_512
        }

        public static Dictionary<WarpType, torch.Tensor> WarpTemplates = new Dictionary<WarpType, Tensor> {
            {WarpType.Arcface_112_v1, torch.tensor(
                new float[]{ 0.35473214f, 0.45658929f,0.64526786f, 0.45658929f,
                    0.50000000f,0.61154464f,0.37913393f, 0.77687500f,
                    0.62086607f, 0.77687500f}).reshape(5,2) },
            {WarpType.Arcface_112_v2, torch.tensor(
                new float[]{ 0.34191607f, 0.46157411f, 0.65653393f, 0.45983393f,
                    0.50022500f, 0.64050536f, 0.37097589f, 0.82469196f,
                    0.63151696f, 0.82325089f }).reshape(5, 2) },
            {WarpType.Arcface_128_v2, torch.tensor(
                new float[] { 0.36167656f, 0.40387734f, 0.63696719f, 0.40235469f,
                    0.50019687f, 0.56044219f, 0.38710391f, 0.72160547f,
                    0.61507734f, 0.72034453f }).reshape(5, 2) },
            {WarpType.FFHQ_512, torch.tensor(
                new float[] { 0.37691676f, 0.46864664f, 0.62285697f, 0.46912813f,
                    0.50123859f, 0.61331904f, 0.39308822f, 0.72541100f,
                    0.61150205f, 0.72490465f }).reshape(5, 2) }
        };

        /// <summary>
        /// cv2.estimateAffinePartial2D
        /// </summary>
        /// <param name="src"></param>
        /// <param name="dst"></param>
        /// <returns></returns>
        public static torch.Tensor EstimateAffinePartial2DTorch(torch.Tensor src, torch.Tensor dst)
        {
            int num = (int)src.shape[0];
            int dim = (int)src.shape[1];
            var src_mean = torch.mean(src, new long[] { 0 });
            var dst_mean = torch.mean(dst, new long[] { 0 });
            var src_demean = src - src_mean;
            var dst_demean = dst - dst_mean;
            var A = torch.matmul(dst_demean.t(), src_demean) / num;
            var d = torch.ones(new long[] { dim }, dtype: torch.float32);
            if (torch.linalg.det(A).item<float>() < 0)
            {
                d[dim - 1] = -1;
            }
            var T = torch.eye((dim + 1), dtype: torch.float32);
            var (U, S, V) = torch.linalg.svd(A);
            var rank = torch.linalg.matrix_rank(A).item<Int64>();

            if (rank == 0)
            {
                return torch.zeros_like(T);
            }
            else if (rank == dim - 1)
            {
                if ((torch.linalg.det(U) * torch.linalg.det(V)).item<float>() > 0)
                {
                    T[TensorIndex.Slice(null, dim), TensorIndex.Slice(null, dim)] = torch.matmul(U, V);
                }
                else
                {
                    var s1 = d[dim - 1];
                    d[torch.TensorIndex.Single(dim - 1)] = -1;
                    T[TensorIndex.Slice(null, dim), TensorIndex.Slice(null, dim)] = torch.matmul(torch.matmul(U, torch.diag(d)), V);
                    d[torch.TensorIndex.Single(dim - 1)] = s1;
                }
            }
            else
            {
                T[TensorIndex.Slice(null, dim), TensorIndex.Slice(null, dim)] = torch.matmul(torch.matmul(U, torch.diag(d)), V);
            }
            var scale = 1.0f / src_demean.var(dim: 0, unbiased: false).sum() * (torch.matmul(S, d));

            T[TensorIndex.Slice(null, dim), dim] = dst_mean - scale * (torch.matmul(T[TensorIndex.Slice(null, dim), TensorIndex.Slice(null, dim)], src_mean.t()));
            T[TensorIndex.Slice(null, dim), TensorIndex.Slice(null, dim)] *= scale;
            return T[TensorIndex.Slice(null, 2)];
        }

        public static torch.Tensor EstimateMatrixByFaceLandmark5(torch.Tensor facelandmark5, torch.Tensor normedWarpTemplate)
        {
            var affineMatrix = EstimateAffinePartial2DTorch(facelandmark5, normedWarpTemplate);
            return affineMatrix;
        }

        public static torch.Tensor ConvertAffinematrixToHomography(torch.Tensor M)
        {
            var H = torch.nn.functional.pad(M, new long[] { 0, 0, 0, 1 }, PaddingModes.Constant, 0);
            H[TensorIndex.Ellipsis, -1, -1] += 1.0;
            return H;
        }

        public static torch.Tensor NormalTransformPixel(int height, int width)
        {
            var trMat = torch.tensor(new float[] { 1f, 0f, -1f, 0f, 1f, -1f, 0f, 0f, 1f }).reshape(3, 3);
            float widthDenom, heightDenom;
            if (width == 1)
            {
                widthDenom = (float)1e-14;
            }
            else
            {
                widthDenom = width - 1;
            }
            if (height == 1)
            {
                heightDenom = (float)1e-14;
            }
            else
            {
                heightDenom = height - 1;
            }
            trMat[0, 0] = trMat[0, 0] * 2.0f / widthDenom;
            trMat[1, 1] = trMat[1, 1] * 2.0f / heightDenom;
            return trMat.unsqueeze(0);
        }

        public static torch.Tensor NormalizeHomography(torch.Tensor dstPixTransSrcPix, torch.Size dsizeSrc, torch.Size dsizeDst)
        {
            var srcH = dsizeSrc[0];
            var srcW = dsizeSrc[1];
            var dstH = dsizeDst[0];
            var dstW = dsizeDst[1];
            var srcNormTransSrcPix = NormalTransformPixel((int)srcH, (int)srcW);
            var dstNormTransDstPix = NormalTransformPixel((int)dstH, (int)dstW);
            var srcPixTransSrcNorm = torch.linalg.inv(srcNormTransSrcPix);
            var dstNormTransSrcNorm = dstNormTransDstPix.matmul(dstPixTransSrcPix.matmul(srcPixTransSrcNorm));
            return dstNormTransSrcNorm;
        }

        /// <summary>
        /// cv2.warpAffine
        /// </summary>
        /// <param name="src">[1,C,H,W] float32</param>
        /// <param name="M">[1,2,3] float32</param>
        public static torch.Tensor WarpAffineTorch(torch.Tensor src, torch.Tensor M, torch.Size dstSize)
        {
            int H = (int)src.shape[2];
            int W = (int)src.shape[3];
            var matrix3x3 = ConvertAffinematrixToHomography(M);
            var dstNormTransSrcNorm = NormalizeHomography(matrix3x3, new torch.Size(new int[] { H, W }), dstSize);
            var srcNormTransDstNorm = torch.linalg.inv(dstNormTransSrcNorm);
            var grid = torch.nn.functional.affine_grid(srcNormTransDstNorm[TensorIndex.Colon, TensorIndex.Slice(null, 2), TensorIndex.Colon], size: new long[] { 1, 3, dstSize[0], dstSize[1] }, align_corners: true);
            var warpSrc = torch.nn.functional.grid_sample(src, grid, padding_mode: GridSamplePaddingMode.Border, align_corners: true);
            return warpSrc;
        }

        /// <summary>
        /// cv2.invertAffineTransform
        /// </summary>
        /// <param name="M">[1,2,3]</param>
        /// <returns></returns>
        public static torch.Tensor InverseAffineTransformTorch(torch.Tensor M)
        {
            var invMat = torch.zeros_like(M);
            var div1 = M[TensorIndex.Colon, 0, 0] * M[TensorIndex.Colon, 1, 1] - M[TensorIndex.Colon, 0, 1] * M[TensorIndex.Colon, 1, 0];
            invMat[TensorIndex.Colon, 0, 0] = M[TensorIndex.Colon, 1, 1] / div1;
            invMat[TensorIndex.Colon, 0, 1] = -M[TensorIndex.Colon, 0, 1] / div1;
            invMat[TensorIndex.Colon, 0, 2] = -(M[TensorIndex.Colon, 0, 2] * M[TensorIndex.Colon, 1, 1] - M[TensorIndex.Colon, 0, 1] * M[TensorIndex.Colon, 1, 2]) / div1;
            var div2 = M[TensorIndex.Colon, 0, 1] * M[TensorIndex.Colon, 1, 0] - M[TensorIndex.Colon, 0, 0] * M[TensorIndex.Colon, 1, 1];
            invMat[TensorIndex.Colon, 1, 0] = M[TensorIndex.Colon, 1, 0] / div2;
            invMat[TensorIndex.Colon, 1, 1] = -M[TensorIndex.Colon, 0, 0] / div2;
            invMat[TensorIndex.Colon, 1, 2] = -(M[TensorIndex.Colon, 0, 2] * M[TensorIndex.Colon, 1, 0] - M[TensorIndex.Colon, 0, 0] * M[TensorIndex.Colon, 1, 2]) / div2;
            return invMat;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="img">[1,3,H,W] float32</param>
        /// <param name="facelandmark5">float32</param>
        /// <param name="type"></param>
        /// <param name="cropSize"></param>
        /// <returns></returns>
        public static (torch.Tensor, torch.Tensor) WarpFaceByFaceLandmark5(torch.Tensor img, torch.Tensor facelandmark5, WarpType type, torch.Size cropSize)
        {
            torch.Tensor cropSizeTensor = torch.tensor(cropSize.ToArray());
            torch.Tensor normedWarpTemplate = WarpTemplates[type] * cropSizeTensor;
            var affineMatrix = EstimateMatrixByFaceLandmark5(facelandmark5, normedWarpTemplate);
            var warpSrc = WarpAffineTorch(img, affineMatrix, cropSize);
            return (warpSrc, affineMatrix);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="src">[1,3,H,W]</param>
        /// <param name="cropSrc"></param>
        /// <param name="cropMask">[1,1,H,W]</param>
        /// <param name="matrix">[1,2,3]</param>
        /// <returns></returns>
        public static torch.Tensor PasteBack(torch.Tensor src, torch.Tensor cropSrc, torch.Tensor cropMask, torch.Tensor matrix)
        {
            var inverseMatrix = InverseAffineTransformTorch(matrix);
            var dstSize = new torch.Size(new long[] { src.shape[2], src.shape[3] });
            var inverseSrc = WarpAffineTorch(cropSrc, inverseMatrix, dstSize);
            var inverseMask = WarpAffineTorch(cropMask, inverseMatrix, dstSize);
            inverseMask = inverseMask.clip(0, 1).squeeze();
            var pasteSrc = src.clone();
            pasteSrc[TensorIndex.Colon, 0] = inverseMask * inverseSrc[TensorIndex.Colon, 0] + (1.0f - inverseMask) * src[TensorIndex.Colon, 0];
            pasteSrc[TensorIndex.Colon, 1] = inverseMask * inverseSrc[TensorIndex.Colon, 1] + (1.0f - inverseMask) * src[TensorIndex.Colon, 1];
            pasteSrc[TensorIndex.Colon, 2] = inverseMask * inverseSrc[TensorIndex.Colon, 2] + (1.0f - inverseMask) * src[TensorIndex.Colon, 2];
            return pasteSrc;
        }

        public static torch.Tensor BlendImg(torch.Tensor src, torch.Tensor paste)
        {
            float ratio = 0.2f;
            var result = src * ratio + paste * (1 - ratio);
            return result;
        }

        public static torch.Tensor CreateBoxMask(long height, long width)
        {
            var cropMask = torch.ones(new long[] { 1, 1, height, width }, dtype: torch.float32);
            cropMask[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(null, 20), TensorIndex.Colon] = 0;
            cropMask[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(-20, null), TensorIndex.Colon] = 0;
            cropMask[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(null, 20)] = 0;
            cropMask[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(-20, null)] = 0;
            long kernel = 21;
            var trans = torchvision.transforms.GaussianBlur(kernel, kernel, 10, 11);
            cropMask = trans.call(cropMask);
            return cropMask;
        }

        public static torch.Tensor Preprocess(torch.Tensor input)
        {
            input = input / 255f;
            input = (input - 0.5f) / 0.5f;
            input = torch.clip(input, -1f, 1f);
            return input;
        }

        public static torch.Tensor Postprocess(torch.Tensor input)
        {
            input = torch.clip(input, -1f, 1f);
            input = (input + 1f) / 2f;
            input = (input * 255).round();
            return input;
        }
    }
}
