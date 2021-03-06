#include "Denoise.h"


void NLM::NonLocalMeans(int Ds, int ds)
{
	cv::Mat srcImg = cv::imread("lena.jpg");
	cv::Mat grayImg;
	cv::Mat dst;
	cv::cvtColor(srcImg, grayImg, cv::COLOR_BGR2GRAY);
	dst = grayImg.clone();

	int Height = grayImg.rows;
	int Width = grayImg.cols;
	std::cout << "rows: " << Height << ", cols: " << Width << std::endl;
	cv::imshow("srcImg", srcImg);
	cv::imshow("grayImg", grayImg);
	float h = 5;
	//////////////////////////////////////////////////////////////
	///
	///
	///
	/////////////////////////////////////////////////////////////
	float kernelValue = 1 / ((2 * ds + 1) * (2 * ds + 1));
	std::vector<std::vector<float> > Weight;
	Weight.resize(2 * (Ds-ds)+1);
	for (int k = 0; k < 2 * (Ds - ds) + 1; k++)
	{
		Weight[k].resize(2 * (Ds - ds) + 1);
	}
	for (int k = 0; k < 2 * (Ds - ds) + 1; k++)
	{
		for (int j = 0; j < 2 * (Ds - ds) + 1; j++)
		{
			Weight[k][j] = 0;
		}
	}



	for (int row = Ds; row < Height - Ds; row++)
	{
		for (int col = Ds; col < Width - Ds; col++)
		{
			//确定block的中心
			for (int SearchWindowsX = -Ds+ds; SearchWindowsX < Ds-ds; SearchWindowsX++)
			{
				for (int SearchWindowsY = -Ds+ds; SearchWindowsY < Ds-ds; SearchWindowsY++)
				{
					float distance = 0;
					//block中心
					int blockCenterX = row + SearchWindowsX;
					int blockCenterY = col + SearchWindowsY;
					float sum = 0;
					for (int i = -ds; i < ds; i++)
					{
						for (int j = -ds; j < ds; j++)
						{
							distance += kernelValue * pow((grayImg.at<uchar>(blockCenterX + i, blockCenterY + j) - grayImg.at<uchar>(row + i, col+j)), 2);
						}
					}
					Weight[SearchWindowsX + Ds-ds][SearchWindowsY + Ds-ds] = pow(2.718, (-distance / (h * h)));

				}
			}
			float max = 0.0;
			float sum = 0.0;

			//找出weight中的最大值赋给中心且求出和以进行归一化
			for (int i = 0; i < 2 * (Ds - ds) + 1; i++)
			{
				for (int j = 0; j < 2 * (Ds - ds) + 1; j++)
				{
					if (Weight[i][j]>max)
					{
						max = Weight[i][j];
					}
					sum += Weight[i][j];
				}
			}
			Weight[Ds - ds + 1][Ds - ds + 1] = max;
			//归一化权重并求得
			float PixelSum = 0.0;
			for (int i = 0; i < 2 * (Ds - ds) + 1; i++)
			{
				for (int j = 0; j < 2 * (Ds - ds) + 1; j++)
				{
					Weight[i][j] /= sum;
					PixelSum += grayImg.at<uchar>(row - Ds + i, col - Ds + j) * Weight[i][j];
				}
			}
			dst.at<uchar>(row, col) = int(PixelSum);
			//std::cout << "row:" << row << ", col: " << col << std::endl;

		}
	}
	cv::imshow("dst", dst);
	cv::waitKey(0);
}


void NLM::NLMWithIntegralImg(cv::Mat IntegralImg,cv::Mat grayImg, int Ds, int ds)
{
	cv::Mat dst = grayImg.clone();

	int Height = grayImg.rows;
	int Width = grayImg.cols;
	std::cout << "rows: " << Height << ", cols: " << Width << std::endl;
	float h = 5;



	double kernelValue = 1 / ((2 * ds + 1) * (2 * ds + 1));
	std::cout << "kernelValue: " << kernelValue << std::endl;

	std::vector<std::vector<float> > Weight;
	Weight.resize(2 * (Ds - ds) + 1);
	for (int k = 0; k < 2 * (Ds - ds) + 1; k++)
	{
		Weight[k].resize(2 * (Ds - ds) + 1);
	}
	for (int k = 0; k < 2 * (Ds - ds) + 1; k++)
	{
		for (int j = 0; j < 2 * (Ds - ds) + 1; j++)
		{
			Weight[k][j] = 0;
		}
	}



	for (int row = Ds; row < Height - Ds; row++)
	{
		for (int col = Ds; col < Width - Ds; col++)
		{
			//确定block的中心
			float max = 0;
			float sum = 0;
			for (int SearchWindowsX = -Ds + ds; SearchWindowsX < Ds - ds; SearchWindowsX++)
			{
				for (int SearchWindowsY = -Ds + ds; SearchWindowsY < Ds - ds; SearchWindowsY++)
				{
					float distance = 0;
					//block中心
					int blockCenterX = row + SearchWindowsX;
					int blockCenterY = col + SearchWindowsY;

					int blockLUPouintX = row - ds;
					int blockLUPouintY = col - ds;
					int blockRDPouintX = row + ds;
					int blockRDPouintY = col + ds;

					float blockValue = IntegralImg.at<float>(blockRDPouintX, blockRDPouintY) + IntegralImg.at<float>(blockLUPouintX, blockLUPouintY)- IntegralImg.at<float>(blockLUPouintX, blockRDPouintY) - IntegralImg.at<float>(blockRDPouintX, blockLUPouintY);
					float slipValue = IntegralImg.at<float>(blockCenterX + ds, blockCenterY + ds)- IntegralImg.at<float>(blockCenterX + ds, blockCenterY - ds) - IntegralImg.at<float>(blockCenterX - ds, blockCenterY + ds) + IntegralImg.at<float>(blockCenterX - ds, blockCenterY - ds);


					distance = kernelValue * pow((blockValue - slipValue), 2);
					//std::cout << "distance: " << distance << std::endl;
					//std::cout << "kernelValue: " << kernelValue << std::endl;

					Weight[SearchWindowsX + Ds - ds][SearchWindowsY + Ds - ds] = pow(2.718, (-distance / (h * h)));
					//std::cout << "Weight: " << Weight[SearchWindowsX + Ds - ds][SearchWindowsY + Ds - ds] << std::endl;

					if (Weight[SearchWindowsX + Ds - ds][SearchWindowsY + Ds - ds]>max)
					{
						max = Weight[SearchWindowsX + Ds - ds][SearchWindowsY + Ds - ds];
					}

					sum += Weight[SearchWindowsX + Ds - ds][SearchWindowsY + Ds - ds];
					//std::cout << "sum: " << sum << std::endl;

				}
			}
			Weight[Ds - ds + 1][Ds - ds + 1] = max;
			//std::cout << "sum: " << sum << std::endl;
			//std::cout << "Weight: " << Weight[Ds - ds + 1][Ds - ds + 1] << std::endl;

			//归一化权重并求得
			float PixelSum = 0.0;
			for (int i = 0; i < 2 * (Ds - ds) + 1; i++)
			{
				for (int j = 0; j < 2 * (Ds - ds) + 1; j++)
				{
					//Weight[i][j] /= sum;
					PixelSum += grayImg.at<uchar>(row - Ds + i, col - Ds + j) * Weight[i][j];

				}
			}
			PixelSum /= sum;
			//std::cout << "PixelSum: " << PixelSum << std::endl;
			//if (PixelSum > 255) std::cout << "PixelSum: " << PixelSum << std::endl;
			//PixelSum > 255 ? PixelSum = 255 : PixelSum=PixelSum;
			dst.at<uchar>(row, col) = int(PixelSum);
			//std::cout << "row:" << row << ", col: " << col << std::endl;
		}
	}
	cv::imshow("dst", dst);
	cv::waitKey(0);
}
© 2021 GitHub, Inc.
Terms
Privacy
Security
Status

