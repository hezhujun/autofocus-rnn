#include <fstream>
#include <iostream>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

#define M_PI 3.14159265358979323846

inline double f(Mat& image, int x, int y)
{
	uchar* element = image.ptr<uchar>(x, y);
	return (double)(*element);
}

double Brenner(Mat& image)
{
	double focusMeasure = 0;
	for (int x = 0; x <= image.rows - 1; x++)
	{
		for (int y = 0; y <= image.cols - 3; y++)
		{
			double diff = f(image, x, y + 2) - f(image, x, y);
			focusMeasure += diff * diff;
			if (focusMeasure < 0)
			{
				cout << "error: overflow in Brenner" << endl;
			}
		}
	}
	return focusMeasure;
}

double verticalBrenner(Mat& image)
{
	double focusMeasure = 0;
	for (int x = 0; x <= image.rows - 3; x++)
	{
		for (int y = 0; y <= image.cols - 1; y++)
		{
			double diff = f(image, x + 2, y) - f(image, x, y);
			focusMeasure += diff * diff;
			if (focusMeasure < 0)
			{
				cout << "error: overflow in vertical Brenner" << endl;
			}
		}
	}
	return focusMeasure;
}

double squaredGradient(Mat& image)
{
	double focusMeasure = 0;
	for (int x = 0; x <= image.rows - 1; x++)
	{
		for (int y = 0; y <= image.cols - 2; y++)
		{
			double diff = f(image, x, y + 1) - f(image, x, y);
			focusMeasure += diff * diff;
			if (focusMeasure < 0)
			{
				cout << "error: overflow in squared gradient" << endl;
			}
		}
	}
	return focusMeasure;
}

double verticalSquaredGradient(Mat& image)
{
	double focusMeasure = 0;
	for (int x = 0; x <= image.rows - 2; x++)
	{
		for (int y = 0; y <= image.cols - 1; y++)
		{
			double diff = f(image, x + 1, y) - f(image, x, y);
			focusMeasure += diff * diff;
			if (focusMeasure < 0)
			{
				cout << "error: overflow in vertical squared gradient" << endl;
			}
		}
	}
	return focusMeasure;
}

double _3_3_difference(Mat& image)
{
	double focusMeasure = 0;
	for (int x = 1; x <= image.rows - 2; x++)
	{
		for (int y = 1; y <= image.cols - 2; y++)
		{
			double horizontal = 
				 0 * f(image, x - 1, y - 1) + 0 * f(image, x - 1, y) +  0 * f(image, x - 1, y + 1) +
				-1 * f(image, x, y - 1)     + 0 * f(image, x, y)     + +1 * f(image, x, y + 1)     +
				 0 * f(image, x + 1, y - 1) + 0 * f(image, x + 1, y) +  0 * f(image, x + 1, y + 1);
			double vertical = 
				0 * f(image, x - 1, y - 1) + -1 * f(image, x - 1, y) + 0 * f(image, x - 1, y + 1) +
				0 * f(image, x, y - 1)     +  0 * f(image, x, y)     + 0 * f(image, x, y + 1)     +
				0 * f(image, x + 1, y - 1) + +1 * f(image, x + 1, y) + 0 * f(image, x + 1, y + 1);

			focusMeasure += horizontal * horizontal + vertical * vertical;
			if (focusMeasure < 0)
			{
				cout << "error: overflow in 3*3 difference" << endl;
			}
		}
	}
	return focusMeasure;
}

double _3_3_Sobel(Mat& image)
{
	double focusMeasure = 0;
	for (int x = 1; x <= image.rows - 2; x++)
	{
		for (int y = 1; y <= image.cols - 2; y++)
		{
			double horizontal =
				-1 * f(image, x - 1, y - 1) + 0 * f(image, x - 1, y) + +1 * f(image, x - 1, y + 1) +
				-2 * f(image, x, y - 1)     + 0 * f(image, x, y)     + +2 * f(image, x, y + 1)     +
				-1 * f(image, x + 1, y - 1) + 0 * f(image, x + 1, y) + +1 * f(image, x + 1, y + 1);
			double vertical =
				-1 * f(image, x - 1, y - 1) + -2 * f(image, x - 1, y) + -1 * f(image, x - 1, y + 1) +
				 0 * f(image, x, y - 1)     +  0 * f(image, x, y)     +  0 * f(image, x, y + 1)     +
				+1 * f(image, x + 1, y - 1) + +2 * f(image, x + 1, y) + +1 * f(image, x + 1, y + 1);

			focusMeasure += horizontal * horizontal + vertical * vertical;
			if (focusMeasure < 0)
			{
				cout << "error: overflow in 3*3 Sobel" << endl;
			}
		}
	}
	return focusMeasure;
}

double _3_3_Scharr(Mat& image)
{
	double focusMeasure = 0;
	for (int x = 1; x <= image.rows - 2; x++)
	{
		for (int y = 1; y <= image.cols - 2; y++)
		{
			double horizontal =
				 -3 * f(image, x - 1, y - 1) + 0 * f(image, x - 1, y) +  +3 * f(image, x - 1, y + 1) +
				-10 * f(image, x, y - 1)     + 0 * f(image, x, y)     + +10 * f(image, x, y + 1)     +
				 -3 * f(image, x + 1, y - 1) + 0 * f(image, x + 1, y) +  +3 * f(image, x + 1, y + 1);
			double vertical =
				-3 * f(image, x - 1, y - 1) + -10 * f(image, x - 1, y) + -3 * f(image, x - 1, y + 1) +
				 0 * f(image, x, y - 1)     +   0 * f(image, x, y)     +  0 * f(image, x, y + 1)     +
				+3 * f(image, x + 1, y - 1) + +10 * f(image, x + 1, y) + +3 * f(image, x + 1, y + 1);

			focusMeasure += horizontal * horizontal + vertical * vertical;
			if (focusMeasure < 0)
			{
				cout << "error: overflow in 3*3 Scharr" << endl;
			}
		}
	}
	return focusMeasure;
}

double _3_3_Roberts(Mat& image)
{
	double focusMeasure = 0;
	for (int x = 1; x <= image.rows - 2; x++)
	{
		for (int y = 1; y <= image.cols - 2; y++)
		{
			double horizontal =
				 0 * f(image, x - 1, y - 1) +  0 * f(image, x - 1, y) + 0 * f(image, x - 1, y + 1) +
				 0 * f(image, x, y - 1)     + +1 * f(image, x, y)     + 0 * f(image, x, y + 1)     +
				-1 * f(image, x + 1, y - 1) +  0 * f(image, x + 1, y) + 0 * f(image, x + 1, y + 1);
			double vertical =
				0 * f(image, x - 1, y - 1) +  0 * f(image, x - 1, y) +  0 * f(image, x - 1, y + 1) +
				0 * f(image, x, y - 1)     + +1 * f(image, x, y)     +  0 * f(image, x, y + 1)     +
				0 * f(image, x + 1, y - 1) +  0 * f(image, x + 1, y) + -1 * f(image, x + 1, y + 1);

			focusMeasure += horizontal * horizontal + vertical * vertical;
			if (focusMeasure < 0)
			{
				cout << "error: overflow in 3*3 Roberts" << endl;
			}
		}
	}
	return focusMeasure;
}

double _3_3_Prewitt(Mat& image)
{
	double focusMeasure = 0;
	for (int x = 1; x <= image.rows - 2; x++)
	{
		for (int y = 1; y <= image.cols - 2; y++)
		{
			double horizontal =
				-1 * f(image, x - 1, y - 1) + 0 * f(image, x - 1, y) + +1 * f(image, x - 1, y + 1) +
				-1 * f(image, x, y - 1)     + 0 * f(image, x, y)     + +1 * f(image, x, y + 1)     +
				-1 * f(image, x + 1, y - 1) + 0 * f(image, x + 1, y) + +1 * f(image, x + 1, y + 1);
			double vertical =
				-1 * f(image, x - 1, y - 1) + -1 * f(image, x - 1, y) + -1 * f(image, x - 1, y + 1) +
				 0 * f(image, x, y - 1)     +  0 * f(image, x, y)     +  0 * f(image, x, y + 1)     +
				+1 * f(image, x + 1, y - 1) + +1 * f(image, x + 1, y) + +1 * f(image, x + 1, y + 1);

			focusMeasure += horizontal * horizontal + vertical * vertical;
			if (focusMeasure < 0)
			{
				cout << "error: overflow in 3*3 Prewitt" << endl;
			}
		}
	}
	return focusMeasure;
}

double _5_5_Sobel(Mat& image)
{
	double focusMeasure = 0;
	for (int x = 2; x <= image.rows - 3; x++)
	{
		for (int y = 2; y <= image.cols - 3; y++)
		{
			double horizontal =
				-1 * f(image, x - 2, y - 2) +  -2 * f(image, x - 1, y - 2) + 0 * f(image, x, y - 2) +  +2 * f(image, x + 1, y - 2) + +1 * f(image, x + 2, y - 2) +
				-4 * f(image, x - 2, y - 1) +  -8 * f(image, x - 1, y - 1) + 0 * f(image, x, y - 1) +  +8 * f(image, x + 1, y - 1) + +4 * f(image, x + 2, y - 1) +
				-6 * f(image, x - 2, y)     + -12 * f(image, x - 1, y)     + 0 * f(image, x, y)     + +12 * f(image, x + 1, y)     + +6 * f(image, x + 2, y)     +
				-4 * f(image, x - 2, y + 1) +  -8 * f(image, x - 1, y + 1) + 0 * f(image, x, y + 1) +  +8 * f(image, x + 1, y + 1) + +4 * f(image, x + 2, y + 1) +
				-1 * f(image, x - 2, y + 2) +  -2 * f(image, x - 1, y + 2) + 0 * f(image, x, y + 2) +  +2 * f(image, x + 1, y + 2) + +1 * f(image, x + 2, y + 2);
			double vertical =
				-1 * f(image, x - 2, y - 2) + -4 * f(image, x - 1, y - 2) +  -6 * f(image, x, y - 2) + -4 * f(image, x + 1, y - 2) + -1 * f(image, x + 2, y - 2) +
				-2 * f(image, x - 2, y - 1) + -8 * f(image, x - 1, y - 1) + -12 * f(image, x, y - 1) + -8 * f(image, x + 1, y - 1) + -2 * f(image, x + 2, y - 1) +
				 0 * f(image, x - 2, y)     +  0 * f(image, x - 1, y)     +   0 * f(image, x, y)     +  0 * f(image, x + 1, y)     +  0 * f(image, x + 2, y)     +
				+2 * f(image, x - 2, y + 1) + +8 * f(image, x - 1, y + 1) + +12 * f(image, x, y + 1) + +8 * f(image, x + 1, y + 1) + +2 * f(image, x + 2, y + 1) +
				+1 * f(image, x - 2, y + 2) + +4 * f(image, x - 1, y + 2) +  +6 * f(image, x, y + 2) + +4 * f(image, x + 1, y + 2) + +1 * f(image, x + 2, y + 2);

			focusMeasure += horizontal * horizontal + vertical * vertical;
			if (focusMeasure < 0)
			{
				cout << "error: overflow in 5*5 Sobel" << endl;
			}
		}
	}
	return focusMeasure;
}

double Gaussian(double x, double y, double sigma) 
{
	return exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
}

double Gaussian_x(double x, double y, double sigma)
{
	return -x / (sigma * sigma) * Gaussian(x, y, sigma);
}

double Gaussian_y(double x, double y, double sigma)
{
	return -y / (sigma * sigma) * Gaussian(x, y, sigma);
}

double _7_7_Gaussian(Mat& image, double sigma = 0.6)
{
	// 构建7*7 高斯一阶算子
	double h[7][7], v[7][7];
	for (int i = 0; i < 7; i++)
	{
		for (int j = 0; j < 7; j++)
		{
			v[6 - i][6 - j] = Gaussian_x(i - 3, j - 3, sigma);
			h[6 - i][6 - j] = Gaussian_y(i - 3, j - 3, sigma);
		}
	}

	double focusMeasure = 0;
	for (int i = 3; i <= image.rows - 4; i++)
	{
		for (int j = 3; j <= image.cols - 4; j++)
		{
			double horizontal =
				h[0][0] * f(image, i - 3, j - 3) +
				h[0][1] * f(image, i - 3, j - 2) +
				h[0][2] * f(image, i - 3, j - 1) +
				h[0][3] * f(image, i - 3, j)     +
				h[0][4] * f(image, i - 3, j + 1) +
				h[0][5] * f(image, i - 3, j + 2) +
				h[0][6] * f(image, i - 3, j + 3) +

				h[1][0] * f(image, i - 2, j - 3) +
				h[1][1] * f(image, i - 2, j - 2) +
				h[1][2] * f(image, i - 2, j - 1) +
				h[1][3] * f(image, i - 2, j)     +
				h[1][4] * f(image, i - 2, j + 1) +
				h[1][5] * f(image, i - 2, j + 2) +
				h[1][6] * f(image, i - 2, j + 3) +

				h[2][0] * f(image, i - 1, j - 3) +
				h[2][1] * f(image, i - 1, j - 2) +
				h[2][2] * f(image, i - 1, j - 1) +
				h[2][3] * f(image, i - 1, j)     +
				h[2][4] * f(image, i - 1, j + 1) +
				h[2][5] * f(image, i - 1, j + 2) +
				h[2][6] * f(image, i - 1, j + 3) +

				h[3][0] * f(image, i, j - 3) +
				h[3][1] * f(image, i, j - 2) +
				h[3][2] * f(image, i, j - 1) +
				h[3][3] * f(image, i, j)     +
				h[3][4] * f(image, i, j + 1) +
				h[3][5] * f(image, i, j + 2) +
				h[3][6] * f(image, i, j + 3) +

				h[4][0] * f(image, i + 1, j - 3) +
				h[4][1] * f(image, i + 1, j - 2) +
				h[4][2] * f(image, i + 1, j - 1) +
				h[4][3] * f(image, i + 1, j)     +
				h[4][4] * f(image, i + 1, j + 1) +
				h[4][5] * f(image, i + 1, j + 2) +
				h[4][6] * f(image, i + 1, j + 3) +

				h[5][0] * f(image, i + 2, j - 3) +
				h[5][1] * f(image, i + 2, j - 2) +
				h[5][2] * f(image, i + 2, j - 1) +
				h[5][3] * f(image, i + 2, j)     +
				h[5][4] * f(image, i + 2, j + 1) +
				h[5][5] * f(image, i + 2, j + 2) +
				h[5][6] * f(image, i + 2, j + 3) +

				h[6][0] * f(image, i + 3, j - 3) +
				h[6][1] * f(image, i + 3, j - 2) +
				h[6][2] * f(image, i + 3, j - 1) +
				h[6][3] * f(image, i + 3, j)     +
				h[6][4] * f(image, i + 3, j + 1) +
				h[6][5] * f(image, i + 3, j + 2) +
				h[6][6] * f(image, i + 3, j + 3);

			double vertical =
				v[0][0] * f(image, i - 3, j - 3) +
				v[0][1] * f(image, i - 3, j - 2) +
				v[0][2] * f(image, i - 3, j - 1) +
				v[0][3] * f(image, i - 3, j)     +
				v[0][4] * f(image, i - 3, j + 1) +
				v[0][5] * f(image, i - 3, j + 2) +
				v[0][6] * f(image, i - 3, j + 3) +

				v[1][0] * f(image, i - 2, j - 3) +
				v[1][1] * f(image, i - 2, j - 2) +
				v[1][2] * f(image, i - 2, j - 1) +
				v[1][3] * f(image, i - 2, j)     +
				v[1][4] * f(image, i - 2, j + 1) +
				v[1][5] * f(image, i - 2, j + 2) +
				v[1][6] * f(image, i - 2, j + 3) +

				v[2][0] * f(image, i - 1, j - 3) +
				v[2][1] * f(image, i - 1, j - 2) +
				v[2][2] * f(image, i - 1, j - 1) +
				v[2][3] * f(image, i - 1, j)     +
				v[2][4] * f(image, i - 1, j + 1) +
				v[2][5] * f(image, i - 1, j + 2) +
				v[2][6] * f(image, i - 1, j + 3) +

				v[3][0] * f(image, i, j - 3) +
				v[3][1] * f(image, i, j - 2) +
				v[3][2] * f(image, i, j - 1) +
				v[3][3] * f(image, i, j)     +
				v[3][4] * f(image, i, j + 1) +
				v[3][5] * f(image, i, j + 2) +
				v[3][6] * f(image, i, j + 3) +

				v[4][0] * f(image, i + 1, j - 3) +
				v[4][1] * f(image, i + 1, j - 2) +
				v[4][2] * f(image, i + 1, j - 1) +
				v[4][3] * f(image, i + 1, j)     +
				v[4][4] * f(image, i + 1, j + 1) +
				v[4][5] * f(image, i + 1, j + 2) +
				v[4][6] * f(image, i + 1, j + 3) +

				v[5][0] * f(image, i + 2, j - 3) +
				v[5][1] * f(image, i + 2, j - 2) +
				v[5][2] * f(image, i + 2, j - 1) +
				v[5][3] * f(image, i + 2, j)     +
				v[5][4] * f(image, i + 2, j + 1) +
				v[5][5] * f(image, i + 2, j + 2) +
				v[5][6] * f(image, i + 2, j + 3) +

				v[6][0] * f(image, i + 3, j - 3) +
				v[6][1] * f(image, i + 3, j - 2) +
				v[6][2] * f(image, i + 3, j - 1) +
				v[6][3] * f(image, i + 3, j)     +
				v[6][4] * f(image, i + 3, j + 1) +
				v[6][5] * f(image, i + 3, j + 2) +
				v[6][6] * f(image, i + 3, j + 3);

			focusMeasure += horizontal * horizontal + vertical * vertical;
			if (focusMeasure < 0)
			{
				cout << "error: overflow in Gaussian" << endl;
			}
		}
	}
	return focusMeasure;
}

double _3_3_Laplacian(Mat& image)
{
	double focusMeasure = 0;
	for (int x = 1; x <= image.rows - 2; x++)
	{
		for (int y = 1; y <= image.cols - 2; y++)
		{
			double tmp =
				-1 * f(image, x - 1, y - 1) + -1 * f(image, x - 1, y) + -1 * f(image, x - 1, y + 1) +
				-1 * f(image, x, y - 1)     + +8 * f(image, x, y)     + -1 * f(image, x, y + 1)     +
				-1 * f(image, x + 1, y - 1) + -1 * f(image, x + 1, y) + -1 * f(image, x + 1, y + 1);

			focusMeasure += tmp * tmp;
			if (focusMeasure < 0)
			{
				cout << "error: overflow in 3*3 Laplacian" << endl;
			}
		}
	}
	return focusMeasure;
}

double _5_5_Laplacian(Mat& image)
{
	double focusMeasure = 0;
	for (int x = 2; x <= image.rows - 3; x++)
	{
		for (int y = 2; y <= image.cols - 3; y++)
		{
			double tmp =
				-1 * f(image, x - 2, y - 2) + -3 * f(image, x - 1, y - 2) +  -4 * f(image, x, y - 2) + -3 * f(image, x + 1, y - 2) + -1 * f(image, x + 2, y - 2) +
				-3 * f(image, x - 2, y - 1) +  0 * f(image, x - 1, y - 1) +  +6 * f(image, x, y - 1) +  0 * f(image, x + 1, y - 1) + -3 * f(image, x + 2, y - 1) +
				-4 * f(image, x - 2, y)     + +6 * f(image, x - 1, y)     + +20 * f(image, x, y)     + +6 * f(image, x + 1, y)     + -4 * f(image, x + 2, y)     +
				-3 * f(image, x - 2, y + 1) +  0 * f(image, x - 1, y + 1) +  +6 * f(image, x, y + 1) +  0 * f(image, x + 1, y + 1) + -3 * f(image, x + 2, y + 1) +
				-1 * f(image, x - 2, y + 2) + -3 * f(image, x - 1, y + 2) +  -4 * f(image, x, y + 2) + -3 * f(image, x + 1, y + 2) + -1 * f(image, x + 2, y + 2);
			
			focusMeasure += tmp * tmp;
			if (focusMeasure < 0)
			{
				cout << "error: overflow in 5*5 Laplacian" << endl;
			}
		}
	}
	return focusMeasure;
}

double _3_3_HorizontalSobel(Mat& image)
{
	double focusMeasure = 0;
	for (int x = 1; x <= image.rows - 2; x++)
	{
		for (int y = 1; y <= image.cols - 2; y++)
		{
			double tmp =
				+1 * f(image, x - 1, y - 1) + +2 * f(image, x - 1, y) + +1 * f(image, x - 1, y + 1) +
				-2 * f(image, x, y - 1)     + -4 * f(image, x, y)     + -2 * f(image, x, y + 1)     +
				+1 * f(image, x + 1, y - 1) + +2 * f(image, x + 1, y) + +1 * f(image, x + 1, y + 1);

			focusMeasure += tmp * tmp;
			if (focusMeasure < 0)
			{
				cout << "error: overflow in 3*3 Horizontal Sobel" << endl;
			}
		}
	}
	return focusMeasure;
}

double _5_5_HorizontalSobel(Mat& image)
{
	double focusMeasure = 0;
	for (int x = 2; x <= image.rows - 3; x++)
	{
		for (int y = 2; y <= image.cols - 3; y++)
		{
			double tmp =
				+1 * f(image, x - 2, y - 2) + +4 * f(image, x - 1, y - 2) +  +6 * f(image, x, y - 2) + +4 * f(image, x + 1, y - 2) + +1 * f(image, x + 2, y - 2) +
				 0 * f(image, x - 2, y - 1) +  0 * f(image, x - 1, y - 1) +   0 * f(image, x, y - 1) +  0 * f(image, x + 1, y - 1) +  0 * f(image, x + 2, y - 1) +
				-2 * f(image, x - 2, y)     + -8 * f(image, x - 1, y)     + -12 * f(image, x, y)     + -8 * f(image, x + 1, y)     + -2 * f(image, x + 2, y)     +
				 0 * f(image, x - 2, y + 1) +  0 * f(image, x - 1, y + 1) +   0 * f(image, x, y + 1) +  0 * f(image, x + 1, y + 1) +  0 * f(image, x + 2, y + 1) +
				+1 * f(image, x - 2, y + 2) + +4 * f(image, x - 1, y + 2) +  +6 * f(image, x, y + 2) + +4 * f(image, x + 1, y + 2) + +1 * f(image, x + 2, y + 2);

			focusMeasure += tmp * tmp;
			if (focusMeasure < 0)
			{
				cout << "error: overflow in 5*5 Horizontal Sobel" << endl;
			}
		}
	}
	return focusMeasure;
}

double _3_3_VerticalSobel(Mat& image)
{
	double focusMeasure = 0;
	for (int x = 1; x <= image.rows - 2; x++)
	{
		for (int y = 1; y <= image.cols - 2; y++)
		{
			double tmp =
				+1 * f(image, x - 1, y - 1) + -2 * f(image, x - 1, y) + +1 * f(image, x - 1, y + 1) +
				+2 * f(image, x, y - 1)     + -4 * f(image, x, y)     + +2 * f(image, x, y + 1)     +
				+1 * f(image, x + 1, y - 1) + -2 * f(image, x + 1, y) + +1 * f(image, x + 1, y + 1);

			focusMeasure += tmp * tmp;
			if (focusMeasure < 0)
			{
				cout << "error: overflow in 3*3 Vertical Sobel" << endl;
			}
		}
	}
	return focusMeasure;
}

double _5_5_VerticalSobel(Mat& image)
{
	double focusMeasure = 0;
	for (int x = 2; x <= image.rows - 3; x++)
	{
		for (int y = 2; y <= image.cols - 3; y++)
		{
			double tmp =
				+1 * f(image, x - 2, y - 2) + 0 * f(image, x - 1, y - 2) +  -2 * f(image, x, y - 2) + 0 * f(image, x + 1, y - 2) + +1 * f(image, x + 2, y - 2) +
				+4 * f(image, x - 2, y - 1) + 0 * f(image, x - 1, y - 1) +  -8 * f(image, x, y - 1) + 0 * f(image, x + 1, y - 1) + +4 * f(image, x + 2, y - 1) +
				+6 * f(image, x - 2, y)     + 0 * f(image, x - 1, y)     + -12 * f(image, x, y)     + 0 * f(image, x + 1, y)     + +6 * f(image, x + 2, y)     +
				+4 * f(image, x - 2, y + 1) + 0 * f(image, x - 1, y + 1) +  -8 * f(image, x, y + 1) + 0 * f(image, x + 1, y + 1) + +4 * f(image, x + 2, y + 1) +
				+1 * f(image, x - 2, y + 2) + 0 * f(image, x - 1, y + 2) +  -2 * f(image, x, y + 2) + 0 * f(image, x + 1, y + 2) + +1 * f(image, x + 2, y + 2);

			focusMeasure += tmp * tmp;
			if (focusMeasure < 0)
			{
				cout << "error: overflow in 5*5 Vertical Sobel" << endl;
			}
		}
	}
	return focusMeasure;
}

double _3_3_CrossSobel(Mat& image)
{
	double focusMeasure = 0;
	for (int x = 1; x <= image.rows - 2; x++)
	{
		for (int y = 1; y <= image.cols - 2; y++)
		{
			double tmp =
				-1 * f(image, x - 1, y - 1) + 0 * f(image, x - 1, y) + +1 * f(image, x - 1, y + 1) +
				 0 * f(image, x, y - 1)     + 0 * f(image, x, y)     +  0 * f(image, x, y + 1)     +
				+1 * f(image, x + 1, y - 1) + 0 * f(image, x + 1, y) + -1 * f(image, x + 1, y + 1);

			focusMeasure += tmp * tmp;
			if (focusMeasure < 0)
			{
				cout << "error: overflow in 3*3 Cross Sobel" << endl;
			}
		}
	}
	return focusMeasure;
}

double _5_5_CrossSobel(Mat& image)
{
	double focusMeasure = 0;
	for (int x = 2; x <= image.rows - 3; x++)
	{
		for (int y = 2; y <= image.cols - 3; y++)
		{
			double tmp =
				-1 * f(image, x - 2, y - 2) + -2 * f(image, x - 1, y - 2) + 0 * f(image, x, y - 2) + +2 * f(image, x + 1, y - 2) + +1 * f(image, x + 2, y - 2) +
				-2 * f(image, x - 2, y - 1) + -4 * f(image, x - 1, y - 1) + 0 * f(image, x, y - 1) + +4 * f(image, x + 1, y - 1) + +2 * f(image, x + 2, y - 1) +
				 0 * f(image, x - 2, y)     +  0 * f(image, x - 1, y)     + 0 * f(image, x, y)     +  0 * f(image, x + 1, y)     +  0 * f(image, x + 2, y)     +
				+2 * f(image, x - 2, y + 1) + +4 * f(image, x - 1, y + 1) + 0 * f(image, x, y + 1) + -4 * f(image, x + 1, y + 1) + -2 * f(image, x + 2, y + 1) +
				+1 * f(image, x - 2, y + 2) + +2 * f(image, x - 1, y + 2) + 0 * f(image, x, y + 2) + -2 * f(image, x + 1, y + 2) + -1 * f(image, x + 2, y + 2);

			focusMeasure += tmp * tmp;
			if (focusMeasure < 0)
			{
				cout << "error: overflow in 5*5 Cross Sobel" << endl;
			}
		}
	}
	return focusMeasure;
}

// Laplacian of Gaussian function
double LOG(double x, double y, double sigma)
{
	return (-2 / (sigma * sigma) + (x * x + y * y) /
		(sigma * sigma * sigma * sigma)) * Gaussian(x, y, sigma);
}

double _9_9_LaplacianOfGaussian(Mat& image, double sigma = 0.6)
{
	// 构建9*9 高斯-拉普拉斯算子
	double h[9][9];
	for (int i = 0; i < 9; i++)
	{
		for (int j = 0; j < 9; j++)
		{
			h[8 - i][8 - j] = Gaussian_x(i - 4, j - 4, sigma);
		}
	}

	double focusMeasure = 0;
	for (int x = 4; x <= image.rows - 5; x++)
	{
		for (int y = 4; y <= image.cols - 5; y++)
		{
			double tmp =
				h[0][0] * f(image, x - 4, y - 4) +
				h[0][1] * f(image, x - 4, y - 3) +
				h[0][2] * f(image, x - 4, y - 2) +
				h[0][3] * f(image, x - 4, y - 1) +
				h[0][4] * f(image, x - 4, y)     +
				h[0][5] * f(image, x - 4, y + 1) +
				h[0][6] * f(image, x - 4, y + 2) +
				h[0][7] * f(image, x - 4, y + 3) +
				h[0][8] * f(image, x - 4, y + 4) +

				h[1][0] * f(image, x - 3, y - 4) +
				h[1][1] * f(image, x - 3, y - 3) +
				h[1][2] * f(image, x - 3, y - 2) +
				h[1][3] * f(image, x - 3, y - 1) +
				h[1][4] * f(image, x - 3, y)     +
				h[1][5] * f(image, x - 3, y + 1) +
				h[1][6] * f(image, x - 3, y + 2) +
				h[1][7] * f(image, x - 3, y + 3) +
				h[1][8] * f(image, x - 3, y + 4) +

				h[2][0] * f(image, x - 2, y - 4) +
				h[2][1] * f(image, x - 2, y - 3) +
				h[2][2] * f(image, x - 2, y - 2) +
				h[2][3] * f(image, x - 2, y - 1) +
				h[2][4] * f(image, x - 2, y)     +
				h[2][5] * f(image, x - 2, y + 1) +
				h[2][6] * f(image, x - 2, y + 2) +
				h[2][7] * f(image, x - 2, y + 3) +
				h[2][8] * f(image, x - 2, y + 4) +

				h[3][0] * f(image, x - 1, y - 4) +
				h[3][1] * f(image, x - 1, y - 3) +
				h[3][2] * f(image, x - 1, y - 2) +
				h[3][3] * f(image, x - 1, y - 1) +
				h[3][4] * f(image, x - 1, y)     +
				h[3][5] * f(image, x - 1, y + 1) +
				h[3][6] * f(image, x - 1, y + 2) +
				h[3][7] * f(image, x - 1, y + 3) +
				h[3][8] * f(image, x - 1, y + 4) +

				h[4][0] * f(image, x, y - 4) +
				h[4][1] * f(image, x, y - 3) +
				h[4][2] * f(image, x, y - 2) +
				h[4][3] * f(image, x, y - 1) +
				h[4][4] * f(image, x, y)     +
				h[4][5] * f(image, x, y + 1) +
				h[4][6] * f(image, x, y + 2) +
				h[4][7] * f(image, x, y + 3) +
				h[4][8] * f(image, x, y + 4) +

				h[5][0] * f(image, x + 1, y - 4) +
				h[5][1] * f(image, x + 1, y - 3) +
				h[5][2] * f(image, x + 1, y - 2) +
				h[5][3] * f(image, x + 1, y - 1) +
				h[5][4] * f(image, x + 1, y)     +
				h[5][5] * f(image, x + 1, y + 1) +
				h[5][6] * f(image, x + 1, y + 2) +
				h[5][7] * f(image, x + 1, y + 3) +
				h[5][8] * f(image, x + 1, y + 4) +

				h[6][0] * f(image, x + 2, y - 4) +
				h[6][1] * f(image, x + 2, y - 3) +
				h[6][2] * f(image, x + 2, y - 2) +
				h[6][3] * f(image, x + 2, y - 1) +
				h[6][4] * f(image, x + 2, y)     +
				h[6][5] * f(image, x + 2, y + 1) +
				h[6][6] * f(image, x + 2, y + 2) +
				h[6][7] * f(image, x + 2, y + 3) +
				h[6][8] * f(image, x + 2, y + 4) +

				h[7][0] * f(image, x + 3, y - 4) +
				h[7][1] * f(image, x + 3, y - 3) +
				h[7][2] * f(image, x + 3, y - 2) +
				h[7][3] * f(image, x + 3, y - 1) +
				h[7][4] * f(image, x + 3, y)     +
				h[7][5] * f(image, x + 3, y + 1) +
				h[7][6] * f(image, x + 3, y + 2) +
				h[7][7] * f(image, x + 3, y + 3) +
				h[7][8] * f(image, x + 3, y + 4) +
				
				h[8][0] * f(image, x + 4, y - 4) +
				h[8][1] * f(image, x + 4, y - 3) +
				h[8][2] * f(image, x + 4, y - 2) +
				h[8][3] * f(image, x + 4, y - 1) +
				h[8][4] * f(image, x + 4, y)     +
				h[8][5] * f(image, x + 4, y + 1) +
				h[8][6] * f(image, x + 4, y + 2) +
				h[8][7] * f(image, x + 4, y + 3) +
				h[8][8] * f(image, x + 4, y + 4);

			focusMeasure += tmp * tmp;
			if (focusMeasure < 0)
			{
				cout << "error: overflow in Laplacian of Gaussian" << endl;
			}
		}
	}
	return focusMeasure;
}

double rangeHistogram(Mat& image)
{
	long h[256] = {0};
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			uchar* pixel = image.ptr<uchar>(i, j);
			h[*pixel]++;
		}
	}

	long maxNumberOfPixel = h[0], minNumberOfPixel = h[0];
	for (int i = 1; i < 256; i++)
	{
		if (h[i] > maxNumberOfPixel)
		{
			maxNumberOfPixel = h[i];
		}
		if (h[i] < minNumberOfPixel)
		{
			minNumberOfPixel = h[i];
		}
	}

	return (double)(maxNumberOfPixel - minNumberOfPixel);
}

double entropyHistogram(Mat& image)
{
	double p[256] = { 0 };
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			uchar* pixel = image.ptr<uchar>(i, j);
			p[*pixel]++;
		}
	}
	double totalPixel = image.rows * image.cols;
	double focusMeasure = 0;
	for (int i = 1; i < 256; i++)
	{
		if (p[i] == 0) continue; // 像素值k的数量为0时不参与计算
		p[i] = p[i] / totalPixel;
		focusMeasure += -p[i] * log2(p[i]);
	}

	return focusMeasure;
}

double M_G_gradient(Mat& image, int i, int j)
{
	double diff1 = f(image, i, j - 1) - f(image, i, j + 1);
	double diff2 = f(image, i - 1, j) - f(image, i + 1, j);
	double diff3 = f(image, i - 1, j - 1) - f(image, i + 1, j + 1);
	double diff4 = f(image, i - 1, j + 1) - f(image, i + 1, j - 1);
	return 2 * diff1 * diff1 + 2 * diff2 * diff2 +
		diff3 * diff3 + diff4 * diff4;
}

double M_G_Histogram(Mat& image)
{
	// 计算阈值theta
	double sum1 = 0, sum2 = 0, gradient;
	for (int i = 1; i < image.rows - 2; i++)
	{
		for (int j = 1; j < image.cols - 2; j++)
		{
			gradient = M_G_gradient(image, i, j);
			sum1 += f(image, i, j) * gradient;
			sum2 += gradient;
		}
	}
	double theta = sum1 / sum2;

	// 计算像素值均值mu
	int mu = (int)(mean(image)[0]);

	// 计算像素值k的数量
	long h[256] = { 0 };
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			uchar* pixel = image.ptr<uchar>(i, j);
			h[*pixel]++;
		}
	}

	double focusMeasure = 0;
	for (int i = mu; i <= 255; i++)
	{
		focusMeasure += (i - theta) * h[i];
	}

	return focusMeasure;
}

double M_M_Histogram(Mat& image)
{
	// 计算像素值均值mu
	int mu = (int)(mean(image)[0]);

	// 计算像素值k的数量
	long h[256] = { 0 };
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			uchar* pixel = image.ptr<uchar>(i, j);
			h[*pixel]++;
		}
	}

	double focusMeasure = 0;
	for (int i = mu; i <= 255; i++)
	{
		focusMeasure += i * h[i];
	}

	return focusMeasure;
}

double variance(Mat& image)
{
	double mu = mean(image)[0];
	Mat tmp;
	image.convertTo(tmp, CV_64F);
	tmp = tmp - mu;
	tmp = tmp.mul(tmp);
	double total = sum(tmp)[0];
	return total / (image.rows * image.cols);
}

double normalizedVariance(Mat& image)
{
	double mu = mean(image)[0];
	Mat tmp;
	image.convertTo(tmp, CV_64F);
	tmp = tmp - mu;
	tmp = tmp.mul(tmp);
	double total = sum(tmp)[0];
	return total / (image.rows * image.cols * mu);
}

double thresholdPixelCount(Mat& image, double threshold = 150)
{
	double sum = 0;
	image.forEach<uchar>([&](uchar& pixel, const int position[]) -> void {
		if (pixel < threshold)
		{
			sum++;
		}
	});
	return sum;
}

double thresholdContent(Mat& image, double threshold = 150)
{
	double sum = 0;
	image.forEach<uchar>([&](uchar& pixel, const int position[]) -> void {
		if (pixel >= threshold)
		{
			sum += pixel;
		}
	});
	return sum;
}

double power(Mat& image, double threshold = 0)
{
	double sum = 0;
	image.forEach<uchar>([&](uchar& pixel, const int position[]) -> void {
		if (pixel >= threshold)
		{
			sum += pixel * pixel;
		}
	});
	return sum;
}

double Vollath_F4(Mat& image)
{
	double item1 = 0, item2 = 0;
	for (int x = 0; x <= image.rows - 2; x++)
	{
		for (int y = 0; y <= image.cols - 1; y++)
		{
			item1 += f(image, x, y) * f(image, x + 1, y);

			if (item1 < 0)
			{
				cout << "error: overflow in Vollath F4" << endl;
			}
		}
	}

	for (int x = 0; x <= image.rows - 3; x++)
	{
		for (int y = 0; y <= image.cols - 1; y++)
		{
			item1 += f(image, x, y) * f(image, x + 2, y);

			if (item2 < 0)
			{
				cout << "error: overflow in Vollath F4" << endl;
			}
		}
	}

	return item1 - item2;
}

double Vollath_F5(Mat& image)
{
	double item1 = 0, item2 = 0;
	for (int x = 0; x <= image.rows - 2; x++)
	{
		for (int y = 0; y <= image.cols - 1; y++)
		{
			item1 += f(image, x, y) * f(image, x + 1, y);

			if (item1 < 0)
			{
				cout << "error: overflow in Vollath F4" << endl;
			}
		}
	}

	double mu = mean(image)[0];
	item2 = image.rows * image.cols * mu * mu;
	
	return item1 - item2;
}

double autocorrelation(Mat& image, int k = 2)
{
	Mat mean, stddev;
	meanStdDev(image, mean, stddev);
	double mu = mean.at<double>(0,0);
	double sigma = stddev.at<double>(0, 0);

	double item1 = (image.rows * image.cols - k) * sigma * sigma;
	double item2 = 0;

	for (int x = 0; x <= image.rows - 1; x++)
	{
		for (int y = 0; y <= image.cols - k - 1; y++)
		{
			item2 += (f(image, x, y) - mu) * (f(image, x, y + k) - mu);
		}
	}

	return item1 - item2;
}

// 直接读取jpeg格式文件的文件大小
double jpeg(char* filepath)
{
	ifstream is;
	is.open(filepath, ios::binary);
	is.seekg(0, ios::end);
	auto length = is.tellg();
	is.close();
	return length;
}

int main(int argc, char* argv[]) 
{
	if (argc < 3)
	{
		cout << "usage: " << argv[0] << " image type [other]" << endl;
		cout << endl;

		cout << "type:" << endl;
		cout << "first derivative" << endl;
		cout << "\t1\tBrenner" << endl;
		cout << "\t2\tvertical Brenner" << endl;
		cout << "\t3\tsquared gradient" << endl;
		cout << "\t4\tvertical squared gradient" << endl;
		cout << "\t5\t3*3 difference" << endl;
		cout << "\t6\t3*3 Sobel" << endl;
		cout << "\t7\t3*3 Scharr" << endl;
		cout << "\t8\t3*3 Roberts" << endl;
		cout << "\t9\t3*3 Prewitt" << endl;
		cout << "\t10\t5*5 Sobel" << endl;
		cout << "\t11\tGaussian" << endl;

		cout << "second derivative" << endl;
		cout << "\t12\tLaplacian of Gaussian" << endl;
		cout << "\t13\t3*3 Laplacian" << endl;
		cout << "\t14\t5*5 Laplacian" << endl;
		cout << "\t15\t3*3 Horizontal Sobel" << endl;
		cout << "\t16\t5*5 Horizontal Sobel" << endl;
		cout << "\t17\t3*3 Vertical Sobel" << endl;
		cout << "\t18\t5*5 Vertical Sobel" << endl;
		cout << "\t19\t3*3 Cross Sobel" << endl;
		cout << "\t20\t5*5 Cross Sobel" << endl;

		cout << "image histogram" << endl;
		cout << "\t21\trange histogram" << endl;
		cout << "\t22\tentropy histogram" << endl;
		cout << "\t23\tM & G histogram" << endl;
		cout << "\t24\tM & M histogram" << endl;

		cout << "image statistics" << endl;
		cout << "\t25\tnormalized variance" << endl;
		cout << "\t26\tvariance" << endl;
		cout << "\t27\tthreshold pixel count" << endl;
		cout << "\t28\tthreshold content" << endl;
		cout << "\t29\tpower" << endl;

		cout << "correlation" << endl;
		cout << "\t30\tautocorrelation" << endl;
		cout << "\t31\tVollath's F4" << endl;
		cout << "\t32\tVollath's F5" << endl;

		cout << "compression" << endl;
		cout << "\t33\tjpeg(directly return file size)" << endl;
		cout << "\t34\tbzip2(not realize)" << endl;
		cout << "\t35\tgzip(not realize)" << endl;
		
		cout << endl;

		cout << "other:" << endl;
		cout << "\tGaussian needs specific sigma, default sigma is 0.6" << endl;
		cout << "\tthreshold pixel count needs a threshold, default threshold is 150" << endl;
		cout << "\tthreshold content needs a threshold, default threshold is 150" << endl;
		cout << "\tpower needs a threshold, default threshold is 0" << endl;
		cout << "\tautocorrelation needs a k, default k is 10" << endl;

		//system("pause");
		return 0;
	}

	Mat image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	if (!image.data)
	{
		cout << argv[1] << " not found" << endl;
		return 0;
	}

	int type = atoi(argv[2]);
	switch (type)
	{
	case 1:
		cout << Brenner(image);
		break;
	case 2:
		cout << verticalBrenner(image);
		break;
	case 3:
		cout << squaredGradient(image);
		break;
	case 4:
		cout << verticalSquaredGradient(image);
		break;
	case 5:
		cout << _3_3_difference(image);
		break;
	case 6:
		cout << _3_3_Sobel(image);
		break;
	case 7:
		cout << _3_3_Scharr(image);
		break;
	case 8:
		cout << _3_3_Roberts(image);
		break;
	case 9:
		cout << _3_3_Prewitt(image);
		break;
	case 10:
		cout << _5_5_Sobel(image);
		break;
	case 11:
	{
		if (argc >= 4)
		{
			double sigma = atof(argv[3]);
			cout << _7_7_Gaussian(image, sigma);
		}
		else
		{
			cout << _7_7_Gaussian(image);
		}
		
	}
		break;
	case 12:
	{
		if (argc >= 4)
		{
			double sigma = atof(argv[3]);
			cout << _9_9_LaplacianOfGaussian(image, sigma);
		}
		else
		{
			cout << _9_9_LaplacianOfGaussian(image);
		}
	}
		break;
	case 13:
		cout << _3_3_Laplacian(image);
		break;
	case 14:
		cout << _5_5_Laplacian(image);
		break;
	case 15:
		cout << _3_3_HorizontalSobel(image);
		break;
	case 16:
		cout << _5_5_HorizontalSobel(image);
		break;
	case 17:
		cout << _3_3_VerticalSobel(image);
		break;
	case 18:
		cout << _5_5_VerticalSobel(image);
		break;
	case 19:
		cout << _3_3_CrossSobel(image);
		break;
	case 20:
		cout << _5_5_CrossSobel(image);
		break;
	case 21:
		cout << rangeHistogram(image);
		break;
	case 22:
		cout << entropyHistogram(image);
		break;
	case 23:
		cout << M_G_Histogram(image);
		break;
	case 24:
		cout << M_M_Histogram(image);
		break;
	case 25:
		cout << normalizedVariance(image);
		break;
	case 26:
		cout << variance(image);
		break;
	case 27:
	{
		if (argc >= 4)
		{
			double threshold = atof(argv[3]);
			cout << thresholdPixelCount(image, threshold);
		}
		else
		{
			cout << thresholdPixelCount(image);
		}
	}
		break;
	case 28:
	{
		if (argc >= 4)
		{
			double threshold = atof(argv[3]);
			cout << thresholdContent(image, threshold);
		}
		else
		{
			cout << thresholdContent(image);
		}
	}
		break;
	case 29:
	{
		if (argc >= 4)
		{
			double threshold = atof(argv[3]);
			cout << power(image, threshold);
		}
		else
		{
			cout << power(image);
		}
	}
		break;
	case 30:
	{
		if (argc >= 4)
		{
			int k = atoi(argv[3]);
			cout << autocorrelation(image, k);
		}
		else
		{
			cout << autocorrelation(image);
		}
	}
		break;
	case 31:
		cout << Vollath_F4(image);
		break;
	case 32:
		cout << Vollath_F5(image);
		break;
	case 33:
	{
		// 检测文件类型
		string filepath(argv[1]);
		size_t dotIndex = filepath.find_last_of('.');
		if (dotIndex < filepath.size())
		{
			string fileExtension = filepath.substr(dotIndex + 1);
			if (fileExtension != "jpg" && fileExtension != "JPG")
			{
				cout << "error: file extension is not \"jpg/JPG\"";
				break;
			}
		}
		else
		{
			cout << "error: file extension is not \"jpg/JPG\"";
			break;
		}
		cout << jpeg(argv[1]);
	}
		break;
	default:
		break;
	}

	//  刷新缓冲区，立刻打印出结果
	cout << flush;

    return 0;
}
