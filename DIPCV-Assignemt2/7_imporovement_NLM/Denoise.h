#ifndef DENOISE_H_INCLUDED
#define DENOISE_H_INCLUDED

#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdio>

class NLM
{
public:
	void NonLocalMeans(int Ds = 5, int ds = 2);
	void NLMWithIntegralImg(cv::Mat IntegralImg,cv::Mat grayImg, int Ds = 5, int ds = 2);
private:

};

#endif // DENOISE_H_INCLUDED
