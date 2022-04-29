// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <vector>

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = MAX_PATH-val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int w = src.step; // no dword alignment is done !!!
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
				/* sau puteti scrie:
				uchar val = lpSrc[i*width + j];
				lpDst[i*width + j] = 255 - val;
				//	w = width pt. imagini cu 8 biti / pixel
				//	w = 3*width pt. imagini cu 24 biti / pixel
				*/
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);
		
		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		int w = src.step; // latimea in octeti a unei linii de imagine
		
		Mat dstH = Mat(height, width, CV_8UC1);
		Mat dstS = Mat(height, width, CV_8UC1);
		Mat dstV = Mat(height, width, CV_8UC1);
		
		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* dstDataPtrH = dstH.data;
		uchar* dstDataPtrS = dstS.data;
		uchar* dstDataPtrV = dstV.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);
		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				// sau int hi = i*w + j * 3;	//w = 3*width pt. imagini 24 biti/pixel
				int gi = i*width + j;
				
				dstDataPtrH[gi] = hsvDataPtr[hi] * 510/360;		// H = 0 .. 255
				dstDataPtrS[gi] = hsvDataPtr[hi + 1];			// S = 0 .. 255
				dstDataPtrV[gi] = hsvDataPtr[hi + 2];			// V = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", dstH);
		imshow("S", dstS);
		imshow("V", dstV);
		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int k = 0.4;
		int pH = 50;
		int pL = k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey();
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey();  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}


// Functii pentru proiect:

float maxim(float a, float b) {
	return a > b ? a : b;
}

float minim(float a, float b) {
	return a < b ? a : b;
}


void RGB2HSV() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, IMREAD_COLOR);
		int height = img.rows;
		int width = img.cols;
		
		Mat red_img = Mat(height, width, CV_8UC3);
		Mat green_img = Mat(height, width, CV_8UC3);
		Mat blue_img = Mat(height, width, CV_8UC3);
		Mat normal_img = Mat(height, width, CV_8UC3);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				Vec3b px = img.at<Vec3b>(i, j);
				unsigned char B = px[0];
				unsigned char G = px[1];
				unsigned char R = px[2];
				
				/*unsigned char red_B = 0;
				unsigned char red_G = 0;
				unsigned char red_R = px[2];

				unsigned char green_B = 0;
				unsigned char green_G = px[1];
				unsigned char green_R = 0;

				unsigned char blue_B = px[0];
				unsigned char blue_G = 0;
				unsigned char blue_R = 0;

				Vec3b red_px = px;
				red_px[0] = 0;
				red_px[1] = 0;

				Vec3b green_px = px;
				green_px[0] = 0;
				green_px[2] = 0;

				Vec3b blue_px = px;
				blue_px[1] = 0;
				blue_px[2] = 0;

				red_img.at<Vec3b>(i, j) = red_px;
				green_img.at<Vec3b>(i, j) = green_px;
				blue_img.at<Vec3b>(i, j) = blue_px;*/
				normal_img.at<Vec3b>(i, j) = px;

				float red_normalized = (float)R / 255;
				float green_normalized = (float)G / 255;
				float blue_normalized = (float)B / 255;

				float M = maxim(red_normalized, green_normalized);
				M = maxim(M, blue_normalized);
				float m = minim(red_normalized, green_normalized);
				m = minim(m, blue_normalized);

				float C = M - m;
				float V; // Vue
				float S; // Saturation
				float H; // Hue
				
				V = M;

				if (V != 0)
					S = C / V;
				else
					S = 0;
				
				if (C != 0) {
					if (M == red_normalized)
						H = 60 * (green_normalized - blue_normalized) / C;
					if (M == green_normalized)
						H = 120 + 60 * (blue_normalized - red_normalized) / C;
					if (M == blue_normalized)
						H = 240 + 60 * (red_normalized - green_normalized) / C;
				}
				else {
					H = 0;
				}

				if (H < 0)
					H = H + 360;

				float H_normalized = H * 255 / 360;
				float S_normalized = S * 255;
				float V_normalized = V * 255;
				Vec3b newPx = px;
				newPx[0] = H_normalized; // H
				newPx[1] = S_normalized; // S
				newPx[2] = V_normalized; // V
				normal_img.at<Vec3b>(i, j) = newPx;
			}
		}

		imshow("input image", img);
		//imshow("R", red_img);
		//imshow("G", green_img);
		//imshow("B", blue_img);
		imshow("HSV?", normal_img);
		waitKey(0);
	}
}

std::vector<float> RGB2HSV_values(void* img, int x, int y) {
	Mat* src = (Mat*)img;
	int height = src->rows;
	int width = src->cols;
	std::vector<float> HSV_vector; // return vector

	Vec3b px = src->at<Vec3b>(x, y);

	unsigned char R = px[2];
	unsigned char G = px[1];
	unsigned char B = px[0];

	float red_normalized = (float)R / 255;
	float green_normalized = (float)G / 255;
	float blue_normalized = (float)B / 255;

	float M = maxim(red_normalized, green_normalized);
	M = maxim(M, blue_normalized);

	float m = minim(red_normalized, green_normalized);
	m = minim(m, blue_normalized);

	float C = M - m;
	float V; // Value
	float S; // Saturation
	float H; // Hue

	V = M;

	if (V != 0)
		S = C / V;
	else
		S = 0;

	if (C != 0) {
		if (M == red_normalized)
			H = 60 * (green_normalized - blue_normalized) / C;
		if (M == green_normalized)
			H = 120 + 60 * (blue_normalized - red_normalized) / C;
		if (M == blue_normalized)
			H = 240 + 60 * (red_normalized - green_normalized) / C;
	}
	else {
		H = 0;
	}

	if (H < 0)
		H = H + 360;


	float H_normalized = H * 255 / 360; // 0..360
	float S_normalized = S * 255;
	float V_normalized = V * 255;

	HSV_vector.push_back(H_normalized);
	HSV_vector.push_back(S_normalized);
	HSV_vector.push_back(V_normalized);

	return HSV_vector;
}

void MyCallBackFunc_HSV(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		Vec3b px = src->at<Vec3b>(y, x);
		std::vector<float> HSV_vector = RGB2HSV_values(src, y, x);
		printf("Ps(x,y) 255 format: %d,%d Color(RGB): %d,%d,%d\n",
			x, y,
			px[2],
			px[1],
			px[0]);
		printf("Ps(x, y) HSV normalized format: %d,%d Color(HSV): %f,%f,%f\n",
			x, y,
			HSV_vector.at(0) * 360 / 255,  // H value -> grades : 0...360
			HSV_vector.at(1) / 255 * 100,  // S value -> percent: 0...100%
			HSV_vector.at(2) / 255 * 100); // V value -> percent: 0...100%
	}
	if(event == CV_EVENT_RBUTTONDOWN)
		RGB2HSV();
}

void testMouseClick_HSV()
{
	Mat src;
	//Read image from file
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc_HSV, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

void RGB2YCbCr() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, IMREAD_COLOR);
		int height = img.rows;
		int width = img.cols;
		Mat return_img = Mat(height, width, CV_8UC3);

		unsigned char delta = 128; // 8 bit images
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				Vec3b px = img.at<Vec3b>(i, j);
				unsigned char B = px[0];
				unsigned char G = px[1];
				unsigned char R = px[2];

				Vec3b processed_px = px;

				/* Trial: bad results

				unsigned char Y = 0.299 * red_px + 0.587 * green_px + 0.114 * blue_px;
				unsigned char Cr = (red_px - Y) * 0.713 + delta;
				unsigned char Cb = (blue_px - Y) * 0.564 + delta;
				red_px = Y + 1.403 * (Cr - delta);
				green_px = Y - 0.714 * (Cr - delta) - 0.344 * (Cb - delta);
				blue_px = Y + 1.773 * (Cb - delta);*/

				unsigned char Y = 16 + 65.738 * R / 256 + 129.057 * G / 256 + 25.064 * B / 256;
				unsigned char Cb = 128 - 37.945 * R / 256 - 74.494 * G / 256 + 112.439 * B / 256;
				unsigned char Cr = 128 + 112.439 * R / 256 - 94.154 * G / 256 - 18.285 * B / 256;

				processed_px[0] = Y;
				processed_px[1] = Cr;
				processed_px[2] = Cb;

				return_img.at<Vec3b>(i, j) = processed_px;
			}
		}
		
		imshow("YCbCr image", return_img);
		waitKey(0);
	}
}

std::vector<float> RGB2YCbCr_values(void* img, int x, int y) {
	Mat* src = (Mat*)img;
	int height = src->rows;
	int width = src->cols;
	std::vector<float> YCbCr_vector; // return vector

	Vec3b px = src->at<Vec3b>(x, y);

	unsigned char R = px[2];
	unsigned char G = px[1];
	unsigned char B = px[0];

	unsigned char delta = 128; // 8 bit images

	/* Trial: bad results
	
	unsigned char Y = 0.299 * R + 0.587 * G + 0.114 * B;
	unsigned char Cr = (R - Y) * 0.713 + delta;
	unsigned char Cb = (B - Y) * 0.564 + delta;*/

	/*unsigned char Y = 0.299 * R + 0.287 * G + 0.11 * B;
	unsigned char Cr = R - Y;
	unsigned char Cb = B - Y;*/

	unsigned char Y  = 16 + 65.738 * R / 256 + 129.057 * G / 256 + 25.064 * B / 256;
	unsigned char Cb = 128 - 37.945 * R / 256 - 74.494 * G / 256 + 112.439 * B / 256;
	unsigned char Cr = 128 + 112.439 * R / 256 - 94.154 * G / 256 - 18.285 * B / 256;

	YCbCr_vector.push_back(Y);
	YCbCr_vector.push_back(Cb);
	YCbCr_vector.push_back(Cr);

	return YCbCr_vector;
}

void MyCallBackFunc_YCbCr(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;

	if (event == CV_EVENT_LBUTTONDOWN)
	{
		Vec3b px = src->at<Vec3b>(y, x);
		std::vector<float> YCbCr_vector = RGB2YCbCr_values(src, y, x);
		printf("Ps(x,y) 255 format: %d,%d Color(RGB): %d,%d,%d\n",
			x, y,
			px[2],
			px[1],
			px[0]);
		printf("Ps(x, y) YCbCr format: %d,%d Color(HSV): %f,%f,%f\n",
			x, y,
			YCbCr_vector.at(0),
			YCbCr_vector.at(1),
			YCbCr_vector.at(2));
	}
	if (event == CV_EVENT_RBUTTONDOWN) {
		RGB2YCbCr();
	}
}

void testMouseClick_YCbCr()
{
	Mat src;
	//Read image from file
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		src = imread(fname);
		// Create a window
		namedWindow("My Window", 1);

		// set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc_YCbCr, &src);

		// show the image
		imshow("My Window", src);

		// wait until user press some key
		waitKey(0);
	}
}

void RGB_values() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, IMREAD_COLOR);
		int height = img.rows;
		int width = img.cols;
		Mat return_img = Mat(height, width, CV_8UC3);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				Vec3b px = img.at<Vec3b>(i, j);
				unsigned char blue_value = px[0];
				unsigned char green_value = px[1];
				unsigned char red_value = px[2];

				if (j == 25) {
					cout << "At px(" << i << ", " << j << "): R = " << (int)red_value << ", G = " << (int)green_value << ", B = " << (int)blue_value << "\n";
					return_img.at<Vec3b>(i, j) = blue_value;
				}
				else {
					return_img.at<Vec3b>(i, j) = px;
				}
			}
		}

		imshow("processed img", return_img);
		waitKey(0);
	}
}


/**
* function to convert RGB image format to HSV, using built - in "cvtColor" function
*/ 
void built_RGB2HSV() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, IMREAD_COLOR);
		Mat hsv;
		cvtColor(img, hsv, COLOR_BGR2HSV);
		imshow("HSV", hsv);
		waitKey(0);
	}
}

/*
* function to convert RGB image format to YCbCr, using built - in "cvtColor" function
*/
void built_RGB2YCbCr() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, IMREAD_COLOR);
		Mat yCbCr;
		cvtColor(img, yCbCr, COLOR_BGR2YCrCb);
		imshow("YCbCr", yCbCr);
		waitKey(0);
	}
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback RGB demo\n");
		printf(" 10 - RGB 2 HSV\n");
		printf(" 11 - RGB 2 YCbCr\n");
		printf(" 12 - RGB values\n");
		printf(" 13 - Mouse callback HSV demo\n");
		printf(" 14 - RGB2HSV built in function\n");
		printf(" 15 - RGB2YCbCr built in function\n");
		printf(" 16 - Mouse callback YCbCr demo\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				RGB2HSV();
				break;
			case 11:
				RGB2YCbCr();
				break;
			case 12:
				RGB_values();
				break;
			case 13:
				testMouseClick_HSV();
				break;
			case 14:
				built_RGB2HSV();
				break;
			case 15:
				built_RGB2YCbCr();
				break;
			case 16:
				testMouseClick_YCbCr();
				break;
		}
	}
	while (op!=0);
	return 0;
}